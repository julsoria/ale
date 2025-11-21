import os
from typing import Optional
import warnings
import torch
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.similarities import SquaredEuclideanDistance
from utils_plus import ProtoPNetSimilarityFormal, _euclidean_distances

###############################################################################
TOP_K = False
MAX_ONLY = False
TRIANGLE_INEQUALITY = False
HYPERSPHERE_APPROXIMATION = not TRIANGLE_INEQUALITY
MAX_EXPLANATIONS = 98_000  # TOTAL = 7*7*2000 = 98,000
DEBUG = False
###############################################################################


class FormalExplanationBase:
    """
    Refactored FormalExplainer focusing on initialization and verification.
    Takes a model and device, handles prototype distances loading/saving,
    and provides a vectorized triangle inequality check.
    """

    def __init__(
        self,
        model: CaBRNet,
        device: Optional[str] = None,
        prototype_filepath: Optional[str] = None,  # Configurable path
        save_proto: bool = False,
        load_proto: bool = True,
        max_explanations: int = MAX_EXPLANATIONS,
        epsilon: float = 1e-5,  # Ideally should be small
    ):
        """
        Initializes the FormalExplainer.

        Args:
            model: The CaBRNet model to explain.
            device: The device to use ('cuda:0', 'cpu', etc.). Autodetects if None.
            prototype_filepath: Path to load/save prototype distances. Required if save_proto=True or load_proto=True.
            save_proto: Whether to calculate and save prototype distances.
            load_proto: Whether to load pre-calculated prototype distances.
            max_explanations: Maximum number of explanations to generate (used by explanation logic).
        """
        print("Initializing Refactored Formal Explanation...")
        
        # --- Argument Validation ---
        if save_proto and load_proto:
            raise ValueError("Cannot save and load prototypes simultaneously.")
        if (save_proto or load_proto) and not prototype_filepath:
            prototype_filepath = os.path.join(os.getcwd(), "prototype_distances.pth")
            warnings.warn("prototype_filepath must be provided if save_proto or load_proto is True.")

        # --- Device Setup ---
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # --- Model Setup ---
        self.model = model
        self.model.eval()
        self.model.to(self.device)

        # --- Model Properties ---
        if not hasattr(model, 'classifier') or \
           not hasattr(model.classifier, 'num_classes') or \
           not hasattr(model.classifier, 'num_prototypes') or \
           not hasattr(model.classifier, 'prototypes') or \
           not hasattr(model.classifier, 'last_layer') or \
           not hasattr(model.classifier, 'similarity_layer') or \
           not hasattr(model.classifier.similarity_layer, 'distances'):
            raise AttributeError("Model does not have the expected structure (classifier, prototypes, similarity_layer.distances, etc.)")

        self.K = self.num_classes = model.classifier.num_classes
        self.P = self.num_prototypes = model.classifier.num_prototypes
        self.prototypes = model.classifier.prototypes.data  # Get data, ensure it's not a parameter requiring grad here
        self.D = self.latent_dim = self.prototypes.shape[1]
        self.weights = model.classifier.last_layer.weight.data  # Shape (K, P)
        # Ensure prototypes are on the correct device for calculations
        self.prototypes = self.prototypes.to(self.device)
        self.epsilon = epsilon
        
        # --- !!! L2-Squared Distance Detection and Adaptation !!! ---
        self.needs_sqrt = False
        proto_dist_calculator_method = self.model.classifier.similarity_layer.distances
        self.use_wrapper = False  # if False, use functions from `extra_utils.py`
        # Check if it's a bound method and if its underlying function is
        # exactly the 'distances' method of the imported SquaredEuclideanDistance class
        if (hasattr(proto_dist_calculator_method, '__func__') and
            proto_dist_calculator_method.__func__ is SquaredEuclideanDistance.distances):
            # This comparison checks if they are the *same function object*
            self.needs_sqrt = True
            if self.use_wrapper:
                print("Detected prototype distance method IS 'SquaredEuclideanDistance.distances'. Applying sqrt wrapper.")
                # Create a wrapper that calculates sqrt(distance) for prototypes
                proto_dist_calculator = _create_sqrt_wrapper(proto_dist_calculator_method, self.epsilon)
            else:
                print("Replacing prototype distance method with L2 distance.")
                # self.model.classifier.similarity_layer = ProtoPNetSimilarityFormal()
                # proto_dist_calculator = self.model.classifier.similarity_layer.distances
                # proto_dist_calculator = EuclideanDistance.distances
                proto_dist_calculator = _euclidean_distances
        else:
            # It's either not a bound method or not the specific function we checked for
            # Could add checks for other scenarios (e.g., direct L2 lambda) if needed
            print("Prototype distance method is NOT 'SquaredEuclideanDistance.distances'. Assuming L2 output.")
            # Use the original calculator directly
            proto_dist_calculator = proto_dist_calculator_method # Assign the original method


        # Define the function to get feature distances (potentially wrapped)
        # (This part remains the same - relies on the needs_sqrt flag)
        original_feature_distance_method = self.model.distances
        if self.needs_sqrt:
            print("Assuming feature distances are also L2-squared. ", {"Applying sqrt wrapper" if self.use_wrapper else "Using L2 replacement method."})
            if self.use_wrapper:
                def sqrt_feature_distance_fn(x):
                    dist_sq = original_feature_distance_method(x)
                    return torch.sqrt(torch.clamp(dist_sq, min=self.epsilon))
                self.feature_distance_fn = sqrt_feature_distance_fn
            else:
                self.model.classifier.similarity_layer = ProtoPNetSimilarityFormal()
                self.feature_distance_fn = self.model.distances
        else:
            print("Assuming model provides L2 distances directly for features.")
            self.feature_distance_fn = original_feature_distance_method
        # --- End of Adaptation ---

        # --- Prototype Distances ---
        self.prototype_distances = None
        loaded_successfully = False
        if load_proto and prototype_filepath and os.path.exists(prototype_filepath):
            try:
                loaded_data = torch.load(prototype_filepath, map_location=self.device)
                print(f"Loaded prototype distances from {prototype_filepath}")
                if self.needs_sqrt:
                    warnings.warn("Assuming loaded prototype distances are L2-squared. Applying sqrt.")
                    self.prototype_distances = torch.sqrt(torch.clamp(loaded_data, min=self.epsilon))
                else:
                    self.prototype_distances = loaded_data

                if self.prototype_distances.shape != (self.P, self.P):
                    warnings.warn(f"Shape mismatch: Expected {(self.P, self.P)}, got {self.prototype_distances.shape}. Recalculating.")
                    self.prototype_distances = None
                    loaded_successfully = False
                else:
                    loaded_successfully = True
            except Exception as e:
                warnings.warn(f"Failed to load prototype distances from {prototype_filepath}: {e}. Will attempt recalculation.")

        if self.prototype_distances is None:
            if load_proto and not loaded_successfully:
                print("Could not load prototype distances. Calculating...")
            else:
                print("Calculating prototype distances...")

            # ** Crucially, NO assert device == "cpu" here **
            with torch.no_grad():
                # Calculate on the designated device
                # Ensure prototypes used for calculation are on the correct device
                protos_for_calc = self.model.classifier.prototypes.to(self.device)
                # Calculate distances using the model's method
                # Assuming it takes (N, D) and (M, D) -> (N, M) or similar that results in (P, P)
                calculated_distances = proto_dist_calculator(protos_for_calc, protos_for_calc)
                
                # Ensure correct shape, detach from graph
                self.prototype_distances = calculated_distances.detach().squeeze()
            
            
            # verify that the diagonal is zero
            # if not, raise error and show which prototype is wrong
            
            # Check if the diagonal is zero
            wrong_prototypes = torch.nonzero(torch.diag(self.prototype_distances) != 0)
            if wrong_prototypes.numel() > 0:
                print(f"Warning: {len(wrong_prototypes)} non-zero diagonal elements detected.")
                for idx in wrong_prototypes:
                    print(f"Prototype {idx.item()} has non-zero distance to itself. d(p, p) = {self.prototype_distances[idx.item(), idx.item()]}")
                    if idx.item() == 20:
                        break
                # print(f"Warning: Non-zero diagonal detected at indices: {wrong_prototypes.flatten().tolist()}")
                # Optionally raise an error or handle this case
                # raise ValueError("Diagonal of prototype distances is not zero. Check distance calculation.")
            assert torch.all(torch.diag(self.prototype_distances) == 0), "Diagonal of prototype distances should be zero (self-distances)."

            if self.prototype_distances.shape != (self.P, self.P):
                raise RuntimeError(f"Calculated prototype distance shape incorrect: Expected {(self.P, self.P)}, got {self.prototype_distances.shape}")

            # Check for and handle negative distances
            negative_mask = self.prototype_distances < 0
            if torch.any(negative_mask):
                num_negative = torch.sum(negative_mask).item()
                warnings.warn(f"Found {num_negative} negative prototype distances. Clamping them to zero. This might indicate an issue with the distance metric.")
                # Clamp negative values to zero
                self.prototype_distances = torch.clamp(self.prototype_distances, min=0.0)

            print("Prototype distances calculated (using L2 distance).")

            if save_proto and prototype_filepath:
                try:
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(prototype_filepath), exist_ok=True)
                    torch.save(self.prototype_distances.cpu(), prototype_filepath) # Save CPU copy
                    print(f"Saved prototype distances to {prototype_filepath}")
                except Exception as e:
                    warnings.warn(f"Failed to save prototype distances to {prototype_filepath}: {e}")

        # Ensure final distances tensor is on the correct device
        self.prototype_distances = self.prototype_distances.to(self.device)
        assert self.prototype_distances.shape == (self.P, self.P), "Final prototype distance shape check failed."
        assert torch.all(self.prototype_distances >= 0), "Negative distances detected after clamping!"


        # --- Other Attributes ---
        self.max_explanations = max_explanations
        self.explanation: list = []  # Use a single clear name
        self.decision_head = self.model.classifier.last_layer  # h (in the original paper)
        self.D = self.latent_dim = self.model.classifier.prototypes.shape[1]  # D
        self.batch_size = 0
        self.batch_idx = -1
        self.explanations_size: list = []
        self.correct_explanations: list = []
        self.incorrect_explanations: list = []
        self.counter_step = 100
        self.H: Optional[int] = None  # Initialize spatial dimensions as None
        self.W: Optional[int] = None  # Will be set when processing first batch

        # Potentially useful, depending on explanation methods
        # self.distance_fn = self.model.distances

        print(f"Initialized with {self.P} prototypes, {self.K} classes. Latent dim: {self.D}.")
        print("Ready for explanation generation.")

    def explain(self, x: torch.Tensor, y: torch.Tensor, verbose: bool = False) -> dict:
        """
        Generate spatial formal explanations for a batch of inputs x with labels y.
        Args:
            x: Input tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels, H is height, and W is width.
            y: True labels of the inputs of shape (N).
            verbose: If True, print additional information.
        
        Returns:
            dict: Explanation dictionary containing pairs of (feature vector, prototype) and their distances.
        """
        # start_batch = time.time()
        # for x_i, y_i in zip(x, y):
        self.explanation = []
        self.explanations_size = []
        self.correct_explanations = []
        self.incorrect_explanations = []
        prediction_corr = (self.model(x)[0].argmax(dim=1) == y)
        explanation = self.explain_one(x, y, verbose=verbose)
        
        self.explanation = []
        if isinstance(explanation, dict):
            # If the explanation is a dictionary, we need to change it to a list
            first_key = list(explanation.keys())[0]  # Get the first key
            if isinstance(first_key, tuple):
                for (l, j), d in explanation.items():
                    self.explanation.append((l, j, d))
            else:
                # If the keys are not tuples, we can just append the values
                self.explanation.extend(explanation.values())
        elif isinstance(explanation, list):
            # If it's a list of explanations, extend the current explanation list
            self.explanation = explanation
        else:
            raise TypeError(f"Unexpected type for explanation: {type(explanation)}. Expected dict or list.")
        # self.explanation.append(explanation)
        # end_batch = time.time()
        # if verbose:
        #     print(f"Generated explanations for {len(x)} inputs.")
        #     print(f"Total time taken for batch: {end_batch - start_batch:.2f} seconds")
        
        # deal with the 'explanations_size' attributes
        exp_size = len(self.explanation)
        self.explanations_size.append(exp_size)
        if prediction_corr:
            self.correct_explanations.append(exp_size)
        else:
            self.incorrect_explanations.append(exp_size)
        
        return self.explanation
    
    def explain_one(self, x: torch.Tensor, y: torch.Tensor, verbose: bool = False) -> dict:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def verify_triangle_inequality(self, x: torch.Tensor, tolerance: float = 1e-6) -> bool:
        """
        Verifies that the triangle inequality holds for distances between
        prototypes and feature vectors using vectorized operations.

        Checks |d(A, C) - d(B, C)| <= d(A, B) <= d(A, C) + d(B, C) for:
            - A, B: Prototypes (from self.prototypes)
            - C: Feature vector (derived from x)

        Args:
            x: Batch of input images, shape (N, C, H_in, W_in).
            tolerance: Numerical tolerance for floating-point comparisons.

        Returns:
            True if the triangle inequality holds for all triplets within
            tolerance, False otherwise.
        """
        # print("Verifying triangle inequality (vectorized)...")
        if self.prototype_distances is None:
            raise RuntimeError("Prototype distances have not been initialized.")

        x = x.to(self.device)
        N = x.size(0)
        # print(f"Input batch size: {N}")
        # --- Check if model is in eval mode ---
        if self.model.training:
            raise RuntimeError("Model must be in eval mode to verify triangle inequality.")
        # --- Check what distance function is used ---
        # print(f"Model distance function: {self.model.distances.__class__}")

        with torch.no_grad():
            # Calculate distances from features to prototypes
            # Expected shape: (N, P, H, W)
            distances_to_features = self.feature_distance_fn(x)  # 
            # distances_to_features = self.model.distances(x)

            # Infer spatial dimensions H, W from the distance map
            _, P_check, H, W = distances_to_features.shape

            # --- Initialization/Validation of H, W ---
            if self.H is None or self.W is None:
                self.H, self.W = H, W
                # print(f"Inferred spatial dimensions H={self.H}, W={self.W}")
            elif self.H != H or self.W != W:
                # This case might occur if the model produces variable output sizes
                print(f"Input batch resulted in different spatial dims ({H}x{W}) than previous ({self.H}x{self.W}). Using current batch's dimensions for this check.")
                # Optionally update self.H, self.W or raise error depending on expected model behavior
                # Sticking with current H, W for this check only:
                # self.H, self.W = H, W

            if P_check != self.P:
                raise ValueError(f"Model distance function returned {P_check} prototype distances, but expected {self.P}")

            # Reshape feature distances for broadcasting: (N, P, H*W)
            d_feat_proto = distances_to_features.reshape(N, self.P, -1)
            num_spatial_locations = H * W

            # Prepare distances for broadcasting:
            # d(p1, p2): (P, P) -> (1, P, P, 1)
            d_p1_p2 = self.prototype_distances.view(1, self.P, self.P, 1)

            # d(p1, z): (N, P, H*W) -> (N, P, 1, H*W)
            d_p1_z = d_feat_proto.unsqueeze(2)

            # d(p2, z): (N, P, H*W) -> (N, 1, P, H*W)
            d_p2_z = d_feat_proto.unsqueeze(1)

            # --- Perform Triangle Inequality Checks ---

            # Check 1: d(p1, p2) <= d(p1, z) + d(p2, z)
            # Broadcasting d_p1_z and d_p2_z yields (N, P, P, H*W)
            # Broadcasting d_p1_p2 yields (N, P, P, H*W)
            check1_valid = (d_p1_p2 <= d_p1_z + d_p2_z + tolerance)

            # Check 2: |d(p1, z) - d(p2, z)| <= d(p1, p2)
            check2_valid = (torch.abs(d_p1_z - d_p2_z) <= d_p1_p2 + tolerance)

            # Combine checks: All must hold for a triplet (n, p1, p2, hw) to be valid
            # Result shape: (N, P, P, H*W)
            all_checks_valid_tensor = check1_valid & check2_valid

            # Check if *all* elements across all dimensions are True
            all_valid = torch.all(all_checks_valid_tensor).item()  # .item() converts 0-dim tensor to Python bool

            # --- Reporting ---
            if all_valid:
                # print("Triangle inequality holds for all checked triplets within tolerance.")
                pass
            else:
                # Count total checks and violations
                # Note: includes p1=p2 cases, which should always be True if tolerance >= 0
                total_checks = N * self.P * self.P * num_spatial_locations
                num_violations = total_checks - torch.sum(all_checks_valid_tensor).item()
                print(f"Triangle inequality VIOLATIONS DETECTED: {num_violations} violations out of {total_checks} checks.")
                print(f"Percentage of violations: {100 * num_violations / total_checks:.2f}%")
                
                # --- Extended Debugging: Find and print first violation details ---
                violating_indices = torch.nonzero(~all_checks_valid_tensor)
                if violating_indices.numel() > 0:
                    # Get indices of the first violation found
                    n_idx, p1_idx, p2_idx, hw_idx = violating_indices[0].tolist()
                    print("-" * 30)
                    print(f"Details for first violation found at:")
                    print(f"  Image index (N): {n_idx}")
                    print(f"  Prototype 1 index (P1): {p1_idx}")
                    print(f"  Prototype 2 index (P2): {p2_idx}")
                    # Calculate H/W coordinates from flattened hw_idx
                    h_idx = hw_idx // W
                    w_idx = hw_idx % W
                    print(f"  Spatial location (H, W): ({h_idx}, {w_idx}) (Flattened index: {hw_idx})")
                    print("-" * 30)

                    # Retrieve the specific scalar distance values involved
                    d_p1_p2_val = self.prototype_distances[p1_idx, p2_idx].item()
                    d_p1_z_val = d_feat_proto[n_idx, p1_idx, hw_idx].item()
                    d_p2_z_val = d_feat_proto[n_idx, p2_idx, hw_idx].item()

                    # Recalculate the bounds for clarity
                    sum_d_pz = d_p1_z_val + d_p2_z_val
                    abs_diff_d_pz = abs(d_p1_z_val - d_p2_z_val)

                    print(f"Values:")
                    print(f"  d(P{p1_idx}, P{p2_idx})       = {d_p1_p2_val:.6f}")
                    print(f"  d(P{p1_idx}, Z)          = {d_p1_z_val:.6f}")
                    print(f"  d(P{p2_idx}, Z)          = {d_p2_z_val:.6f}")
                    print(f"  d(P{p1_idx}, Z) + d(P{p2_idx}, Z) = {sum_d_pz:.6f}")
                    print(f"  |d(P{p1_idx}, Z) - d(P{p2_idx}, Z)| = {abs_diff_d_pz:.6f}")
                    print(f"  Tolerance             = {tolerance}")
                    print("-" * 30)

                    # Check which inequality failed
                    inequality1_holds = (d_p1_p2_val <= sum_d_pz + tolerance)
                    inequality2_holds = (abs_diff_d_pz <= d_p1_p2_val + tolerance)

                    print("Check Results:")
                    if not inequality1_holds:
                        print(f"  FAILED: d(P{p1_idx}, P{p2_idx}) <= d(P{p1_idx}, Z) + d(P{p2_idx}, Z)")
                        print(f"          {d_p1_p2_val:.6f}  >  {sum_d_pz:.6f} + {tolerance}")
                    else:
                        print(f"  PASSED: d(P{p1_idx}, P{p2_idx}) <= d(P{p1_idx}, Z) + d(P{p2_idx}, Z)")

                    if not inequality2_holds:
                        print(f"  FAILED: |d(P{p1_idx}, Z) - d(P{p2_idx}, Z)| <= d(P{p1_idx}, P{p2_idx})")
                        print(f"          {abs_diff_d_pz:.6f}  >  {d_p1_p2_val:.6f} + {tolerance}")
                    else:
                        print(f"  PASSED: |d(P{p1_idx}, Z) - d(P{p2_idx}, Z)| <= d(P{p1_idx}, P{p2_idx})")
                    print("-" * 30)
                # --- End of Extended Debugging ---

        return bool(all_valid)
        # Note: This function does not return the distances, as the main goal is to verify the triangle inequality.

