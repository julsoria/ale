import os
from pathlib import Path

import argparse


import torch
import torch.nn as nn
# import torchinfos
from tqdm import tqdm

from subset_minimal_axp_base import FormalExplanationBase
from utils import check_memory_usage, load_model


def parse_args():
    parser = argparse.ArgumentParser(description="Formal Explanation Arguments")
    parser.add_argument("--top_k", action="store_true", help="Use top-k explanations")
    parser.add_argument("--triangle", action="store_true", help="Use triangle inequality check")
    parser.add_argument("--hypersphere", action="store_true", help="Use hypersphere approximation")
    parser.add_argument("--max_explanations", type=int, default=MAX_EXPLANATIONS, help="Maximum number of explanations")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()
    return args


class SpatialFormalExplanation(FormalExplanationBase):
    """
    A class to generate spatial formal explanations for the prediction of a model.
    Inherits from FormalExplanationBase.
    """

    def __init__(self, model, device="cuda:0", paradigm="triangle", **kwargs):
        super().__init__(model, device=device, **kwargs)
        self.paradigm = paradigm  # Paradigm for the explanation, e.g., "triangle", "hypersphere"
        self.explanation = []  # List to store the explanation
        # Pre-compute and cache frequently used tensors
        self._hw_indices_cache = None
        self._mask_cache = None
        
    def explain_one(self, x, y, verbose=False) -> dict:
        """
        Generate a spatial formal explanation for a single input x with label y.
        Args:
            x: Input tensor.
            y: True label of the input.
            verbose: If True, print additional information.
        """
        
        ### PSEUDO CODE ###
        # -1. Initialize the explainer with the distances with prototypes and the feature vectors.
        # 0. initialize the explanation $E$ by giving, for each feature vector, the closest prototype and the associated similarity score / distance.
        # 1. Forward pass
        # 1._. While there are unverified classes:
        # 1.a. Get (l,j) = NextPair(E) # where l is the feature vector and j is the prototype
        # 1.b. Compute the distance between the feature vector l and the prototype j
        # 1.c. Update the explanation $E$ with the new pair (l,j) and the associated distance
        # 1.d. lb, ub = GenerateBounds(E, paradigm))
        # 1.e. Verify the explanation with the bounds i.e. check how many unverified classes remain
        # 1.f. If there are unverified classes, go back to step 1
        # 2. Backward pass
        # 2._. While the explanation is valid:
        # 2.a. Remove the last pair (l,j) from the explanation *that was not marked as verified* and update the explanation
        # 2.b. lb, ub = GenerateBounds(E, paradigm))
        # 2.c. Verify the explanation with the bounds
        # 2.c.1. If the explanation is still valid, go back to step 2 with the new explanation
        # 2.c.2. If the explanation is not valid, go back to step 2 with the same explanation but mark the last pair as verified
        # 2.c.3. If the explanation is not valid and there are no more pairs to remove, stop the process and return the explanation
        # 3. Return the explanation $E$ with the associated distances and the verified pairs
        ### END PSEUDO CODE ###
        
        # --- Initialization ---
        # Verify triangle inequality before proceeding
        # if verbose:
        #     print(f"Verifying triangle inequality for input distances with tolerance {self.epsilon}...")
        if not self.verify_triangle_inequality(x, tolerance=self.epsilon):
            raise ValueError("Triangle inequality does not hold for input distances. Check distance computation.")
            
        check_memory_usage(threshold_mb=5000)
        
        x = x.to(self.device)  # (N, C, H, W) # images
        y = y.to(self.device)  # (N) # true labels
            
        self.batch_size = x.size(0)  # N
        
        with torch.no_grad():
            # distances_with_proto = self.model.distances(x)  # (N, P, H, W)
            distances_with_proto = self.feature_distance_fn(x)  # (N, P, H, W)
            _, self.num_prototypes, self.H, self.W = distances_with_proto.shape
            # similarities = self.model.similarities(x)  # (N, P, H, W)
            # max_sim, _ = torch.max(similarities, dim=1)  # (P, H*W) -> (P)
            # self.max_sim = max_sim.item()  # Maximum similarity score obtainable
            # similarities_with_proto = self.model.classifier.similarity_layer.distances_to_similarities(distances_with_proto)
            logits = self.model(x)[0]  # (N, K)
            # print(logits.shape)
            y_pred = torch.argmax(logits, dim=1)  # (N)
            y = y_pred.detach().clone()  # Use predicted labels for explanation -- and not true labels !
        # similarities_with_proto = similarities_with_proto.view(batch_size, self.num_prototypes, -1)  # (N, P, H, W) -> (N, P, H*W)
        distances_with_proto = distances_with_proto.view(
            self.batch_size, self.num_prototypes, -1
        )  # (N, P, H, W) -> (N, P, H*W)
        self.distances_with_proto = distances_with_proto
        self._distances_transposed = self.distances_with_proto[0].T  # (H*W, P)
        # min_dis, min_dis_idx = torch.min(distances_with_proto, dim=-1)  # (N, P, H*W) -> (N, P)
        # if the model has the method `features`, we can use it to get the feature vectors, else, use `extractor`
        if hasattr(self.model, "features") and callable(getattr(self.model, "features")):
            # Use the model's features method to get the feature vectors
            self.z = self.model.features(x)
        elif hasattr(self.model, "extractor") and callable(getattr(self.model, "extractor")):
            # Use the model's extractor method to get the feature vectors
            self.z = self.model.extractor(x)
        else:
            raise ValueError(
                "Model does not have a method to extract feature vectors. Please implement 'features' or 'extractor' method."
            )
        assert self.D == self.z.size(1), f"Feature dimension D mismatch: expected {self.D}, got {self.z.size(1)}"
        # torch.stack([min_dis, min_dis_idx], dim=-1) # (N, P, 2)

        # predicted class
        prediction_conf, predicted_class = torch.max(logits.squeeze(), dim=0)  # (N, K) -> (N)
        # predicted_class = torch.argmax(logits, dim=1)  # (N)
        # predicted_class.squeeze()  # (N)
        self.c: int = int(predicted_class.item())  # (1)
        
        # Pre-compute indices and masks for reuse
        self.total_hw = self.H * self.W
        if self._hw_indices_cache is None or len(self._hw_indices_cache) != self.total_hw:
            self._hw_indices_cache = torch.arange(self.total_hw, device=self.device)
        
        E: dict = {}  # Explanation dictionary to store pairs of (feature vector, prototype) and their distances
        E = self._initialize_explanation(x, y)  # Initialize the explanation with the closest prototype for each feature vector
        
        # Sanity checks
        assert self.H is not None and self.W is not None, "Height and width of the feature vectors must be initialized."
        assert self.H > 0 and self.W > 0, "Height and width of the feature vectors must be greater than 0."

        # Forward pass
        unverified_classes = list(range(self.num_classes))  # List of unverified classes
        unverified_classes.remove(self.c)  # Remove the true class from the list of unverified classes
        # progress tracking
        if verbose:
            t = tqdm(total=(self.H * self.W * (self.P - 1)), desc="Forward pass", unit="pair")
            t.n = 0
            t.refresh()
        while unverified_classes:
            # Get the next pair (feature vector, prototype) to explain
            l, j = self._next_pair_round_robin_vectorized(E)
            # if self.paradigm == "hypersphere":
            #     # For hypersphere paradigm, we need to use the estimated centers and radii
            #     # l, j = self._next_pair_round_robin(E)
            #     l, j = self._next_pair_round_robin_vectorized(E)  # from ~6mins to ~12s
            # else:
            #     l, j = self._next_pair(E)
            # Compute the distance between the feature vector l and the prototype j
            distance = self._compute_distance(l, j)
            # Update the explanation E with the new pair (l,j) and the associated distance
            E = self._update_explanation(E, l, j, distance)
            # Generate bounds based on the explanation and the chosen paradigm
            lb, ub = self._generate_bounds(E, paradigm=self.paradigm)
            # Verify the explanation with the bounds
            unverified_classes, unverified_conf = self._verify_explanation(lb, ub, unverified_classes)
            # If there are unverified classes, continue to the next iteration
            if verbose:
                t.n += 1
                postfix_str = f"pred cls: {self.c}, true cls: {y.item()}, conf: {prediction_conf:.2f}, next_l: {l:02d}, next_p: {j:04d}, next_dis: {distance:.2f}, n_unverif: {len(unverified_classes)}, cex conf: {torch.max(unverified_conf):.3f}"
                t.set_postfix_str(postfix_str)
                t.refresh()
            if len(E) >= (self.total_hw * self.num_prototypes):
                # If we have reached the maximum number of explanations, break
                if verbose:
                    print(f"Reached maximum number of explanations: {len(E)}. Stopping forward pass.")
                break
        if verbose:
            t.close()
            print(f"Forward pass completed. Length of explanation E: {len(E)}")
        # Backward pass
        marked_as_verified: set[tuple[int, int]] = set()  # Set to keep track of pairs that were marked as verified
        is_valid = True  # Flag to check if the explanation is still valid
        if verbose:
            t = tqdm(total=len(E), desc="Backward pass", unit="pair")
            t.n = 0
            t.refresh()
        while (len(E) > 1):
            # We check if there are more pairs to remove
            if len(E) == len(marked_as_verified):
                # If there are no more pairs to remove, we stop the process
                break
            # Remove the last pair (l,j) from the explanation that was not marked as verified
            l, j = self._remove_last_pair(E, marked_as_verified)
            # Update the explanation E
            removed_distance = E[(l, j)]
            E = self._update_explanation(E, l, j, None)  # None indicates we are removing the pair
            # Generate bounds based on the updated explanation
            lb, ub = self._generate_bounds(E, paradigm=self.paradigm)
            # Verify the explanation with the bounds
            unverified_classes, unverified_conf = self._verify_explanation(lb, ub, [k for k in range(self.num_classes) if k != self.c])
            is_valid = len(unverified_classes) == 0  # Check if the explanation is still valid
            # print(f"Backward pass: {len(E)} pairs left, {len(unverified_classes)} unverified classes")
            if is_valid:
                # If the explanation is still valid, we continue to the next iteration
                # print(f"Explanation is still valid after removing pair ({l}, {j}). Continuing to the next iteration.\n\tExplanation : {E}")
                # print(f"Bounds: \nlb = \t{lb}\nub = \t{ub}")
                continue
            
            # If the explanation is not valid, we mark the last pair as verified
            E = self._update_explanation(E, l, j, removed_distance)  # Restore the pair with the distance
            marked_as_verified.add((l, j))
            if verbose:
                postfix_str = f"pred cls: {self.c}, true cls: {y.item()}, conf: {prediction_conf:.2f}, mark_l: {l:02d}, mark_p: {j:04d}, next_dis: {distance:.2f}, n_unverif: {len(unverified_classes)}, cex conf: {torch.max(unverified_conf):.3f}"
                t.set_postfix_str(postfix_str)
                t.n += 1
                t.refresh()
                
        # Return the explanation E with the associated distances and the verified pairs
        if verbose:
            t.close()
            print(f"Backward pass completed. Length of explanation E: {len(E)}")
            # print(f"Explanation for input x with label {y.item()}:")
        # print(f"Final explanation E: {E}")
        return E
    
    def _initialize_explanation(self, x, y):
        """
        Initialize the explanation with the closest prototype for each feature vector.
        """
        self.c = y.item()
        
        assert self.H is not None and self.W is not None, "Height and width must be initialized."
        assert self.H > 0 and self.W > 0, "Height and width must be greater than 0."
        
        # Vectorized approach to find the minimum distance for each feature vector
        # self.distances_with_proto has shape (1, P, H*W)
        # We find the min along dimension 1 (prototypes) to get results for each feature vector (H*W)
        min_dists, min_proto_indices = torch.min(self._distances_transposed, dim=1)  # (H*W, P) -> (H*W)
        
        # Create the dictionary from the resulting tensors
        # This is significantly faster than building it inside a loop with computations
        
        # Use pre-computed indices
        E_init = {
            (hw.item(), p.item()): d.item() 
            for hw, p, d in zip(self._hw_indices_cache, min_proto_indices, min_dists)
        }
        
        if self.paradigm == "hypersphere":
            # Initialize estimated centers and radii for the hypersphere approximation
            self.estimated_centers = torch.zeros((self.total_hw, self.D), device=self.device)
            self.estimated_radii = torch.zeros((self.total_hw, 1), device=self.device)
                
            # Vectorized initialization using advanced indexing
            # This gathers the corresponding prototype vectors for all feature vectors at once.
            self.estimated_centers = self.prototypes.squeeze()[(min_proto_indices,)]
            # This assigns the minimum distances as the initial radii.
            self.estimated_radii = min_dists.unsqueeze(1)
        
        return E_init

    def _next_pair(self, E):
        """
        Get the next pair (feature vector, prototype) to explain.
        Args:
            E: Explanation dictionary with pairs of (feature vector, prototype) and their distances.
        Returns:
            tuple: (feature vector index, prototype index)
        """
        
        assert self.H is not None and self.W is not None, "Height and width of the feature vectors must be initialized."
        assert self.H > 0 and self.W > 0, "Height and width of the feature vectors must be greater than 0."
        
        # Create mask more efficiently using advanced indexing
        mask = torch.ones((self.total_hw, self.num_prototypes), dtype=torch.bool, device=self.device)
        if E:
            existing_hw, existing_p = zip(*E.keys())
            existing_hw_tensor = torch.tensor(existing_hw, device=self.device)
            existing_p_tensor = torch.tensor(existing_p, device=self.device)
            mask[existing_hw_tensor, existing_p_tensor] = False
        
        # Find the pair with the minimum distance that is not already in the explanation
        best_pair = None
        if self.paradigm == "triangle":
            # Vectorized distance computation
            distances_masked = self._distances_transposed.clone()  # (H*W, P)
            distances_masked[~mask] = float('inf')  # Mask out existing pairs
            min_idx = torch.argmin(distances_masked)
            best_pair = (min_idx // self.num_prototypes, min_idx % self.num_prototypes)
            
        elif self.paradigm == "hypersphere":
            # The next pair is the one with the maximum absolute cosine value that is not already in the explanation
            # P-C Shape (HW, P, D)
            centers_to_protos = self.prototypes.squeeze((2,3)).squeeze(0) - self.estimated_centers.unsqueeze(1) 
            # print(centers_to_protos.shape)
            
            # Z-C Shape # (H*W, 1, D)
            centers_to_feats = self.z.flatten(start_dim=2).swapaxes(1,2).swapaxes(0,1) - self.estimated_centers.unsqueeze(1)
            # print(centers_to_feats.shape)
            
            # Out: (H*W, P)
            cos = nn.CosineSimilarity(dim=2, eps=1e-8)
            cos_sim = cos(centers_to_protos, centers_to_feats)
            # cos_sim = pairwise_cosine_similarity(estimated_centers)
            # scalar_prods = scalar_prods / (
            #     estimated_radii.view(total_num_feature_vectors, 1) * self.model.classifier.prototypes.norm(dim=1)
            # ) # (H*W, P) / (H*W, 1) * (P,) -> (H*W, P)
            # normalized_scalar_prods = scalar_prods / (torch.norm(estimated_centers, dim=1).view(-1,1) @ torch.norm(self.model.classifier.prototypes.squeeze(), dim=1).view(1,-1))
            # scalar_prods = torch.abs(normalized_scalar_prods)  # (H*W, P)
            scalar_prods = torch.abs(cos_sim)  # (H*W, P)
            # get the hw and p with the smallest scalar product i.e. largest angle + largest distance
            # print(f"scalar_prods shape: {scalar_prods.shape}")
            assert scalar_prods.shape == (self.H * self.W, self.P), f"Scalar products shape: {scalar_prods.shape}"
            
            # --- OPTIMIZATION TO PREVENT GETTING STUCK ---
            # Modify the score to factor in the uncertainty (radius) of each feature vector.
            # This balances refining uncertain regions (large radius) with picking
            # geometrically optimal pairs (high cosine similarity).
            scores = self.estimated_radii * scalar_prods # Shape: (H*W, 1) * (H*W, P) -> (H*W, P)
            # --- END OF MODIFICATION ---

            # Vectorized maximum finding using the new, balanced scores
            scores_masked = scores.clone()
            scores_masked[~mask] = 0.0  # Mask out existing pairs by setting their score to 0
            max_idx = torch.argmax(scores_masked)
            best_pair = (max_idx // self.num_prototypes, max_idx % self.num_prototypes)

        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}. Supported paradigms are 'triangle' and 'hypersphere'.")
        if best_pair is None:
            raise ValueError("No valid pair found for the next explanation step. Check the explanation dictionary and the distances.")
        return best_pair

    def _compute_distance(self, l, j):
        """
        Compute the distance between the feature vector l and the prototype j.
        Args:
            l: Feature vector index.
            j: Prototype index.
        Returns:
            float: Distance between the feature vector and the prototype.
        """
        # l is the feature vector index, j is the prototype index
        distance = self._distances_transposed[l, j].item()
        if distance < 0:
            raise ValueError(f"Computed distance is negative: {distance}. Check the distance computation and the input data.")
        return distance
    
    def _update_explanation(self, E, l, j, distance):
        """
        Update the explanation E with the new pair (l,j) and the associated distance.
        Args:
            E: Explanation dictionary with pairs of (feature vector, prototype) and their distances.
            l: Feature vector index.
            j: Prototype index.
            distance: Distance between the feature vector and the prototype.
        Returns:
            dict: Updated explanation dictionary.
        """
        if distance is None:
            # If distance is None, we are removing the pair from the explanation
            if (l, j) in E:
                del E[(l, j)]
        else:
            # Otherwise, we add or update the pair in the explanation
            E[(l, j)] = distance
        return E
    
    def _remove_last_pair(self, E, marked_as_verified):
        """
        Remove the last pair (l,j) from the explanation that was not marked as verified.
        Args:
            E: Explanation dictionary with pairs of (feature vector, prototype) and their distances.
            marked_as_verified: Set of pairs that were marked as verified.
        Returns:
            tuple: (feature vector index, prototype index) of the removed pair.
        """
        if not E:
            raise ValueError("Explanation is empty. Cannot remove last pair.")
        
        # Get the last pair in the explanation
        E_reversed = reversed(E)
        last_pair = next(E_reversed)  # Get the last pair in the explanation
        
        # If the last pair is already marked as verified, we need to find the next unverified pair
        while last_pair in marked_as_verified:
            # print(f"Skipping verified pair {last_pair}.")
            last_pair = next(E_reversed)  # Skip to the next iteration to find an unverified pair
        # Return the last unverified pair
        l, j = last_pair
        return l, j  # Return the feature vector index and prototype index of the removed pair
    
    def _generate_bounds(self, E, paradigm="triangle") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate bounds based on the explanation E and the chosen paradigm.
        Args:
            E: Explanation dictionary with pairs of (feature vector, prototype) and their distances.
            paradigm: Paradigm for the explanation, e.g., "triangle", "hypersphere".
        Returns:
            tuple: (lower bound, upper bound)
        """
        assert self.H is not None and self.W is not None, "Height and width of the feature vectors must be initialized."
        assert self.H > 0 and self.W > 0, "Height and width of the feature vectors must be greater than 0."
        
        feature_lower_bound_distances = torch.zeros((self.total_hw, self.P), device=self.device)  # (H*W, P)
        feature_upper_bound_distances = torch.ones((self.total_hw, self.P), device=self.device) * float("inf")  # (H*W, P)
        lower_bound_distances = torch.zeros((self.P), device=self.device)  # (P)
        upper_bound_distances = torch.ones((self.P), device=self.device) * float("inf")  # (P)
        
        # distance computations may be erroneous
        warning_counter = 0
        
        # Iterate over the explanation pairs
        if paradigm == "triangle":
            if not E:  # Handle empty explanation
                lower_bound_distances = torch.min(feature_lower_bound_distances, dim=0).values
                upper_bound_distances = torch.min(feature_upper_bound_distances, dim=0).values
                return lower_bound_distances, upper_bound_distances

            # Extract all pairs and distances from E
            if E:
                hw_indices, p_indices = zip(*E.keys())
                distances = list(E.values())
                
                hw_tensor = torch.tensor(hw_indices, device=self.device)
                p_tensor = torch.tensor(p_indices, device=self.device)
                dist_tensor = torch.tensor(distances, device=self.device)
                
                # Set known distances
                feature_lower_bound_distances[hw_tensor, p_tensor] = dist_tensor
                feature_upper_bound_distances[hw_tensor, p_tensor] = dist_tensor
                
                # # Edge-case: the explanation E is the entire set of feature vectors and prototypes
                # if len(E) == self.total_hw * self.P:
                #     # assert that the distances are correct
                #     assert torch.equal(dist_tensor, feature_lower_bound_distances), "Distances in E must match the lower bound distances when E contains all pairs."
                #     assert torch.equal(dist_tensor, feature_upper_bound_distances), "Distances in E must match the upper bound distances when E contains all pairs."
                #     # # If E contains all pairs, lower and upper bound distances must be equal
                #     # feature_lower_bound_distances = dist_tensor.unsqueeze(1).expand_as(feature_lower_bound_distances)
                #     # feature_upper_bound_distances = dist_tensor.unsqueeze(1).expand_as(feature_upper_bound_distances)
                #     # lower_bound_distances = dist_tensor
                #     # upper_bound_distances = dist_tensor
                #     assert torch.equal(
                #         feature_lower_bound_distances, feature_upper_bound_distances
                #     ), "Lower and upper bound distances must be equal when E contains all pairs."
                
                # Vectorized triangle inequality computation
                # Shape: (len(E), P)
                proto_dists_expanded = self.prototype_distances[p_tensor, :]  # (len(E), P)
                dist_expanded = dist_tensor.unsqueeze(1)  # (len(E), 1)
                
                # Compute bounds for all prototype pairs at once
                lb_batch = torch.abs(proto_dists_expanded - dist_expanded)  # (len(E), P)
                ub_batch = proto_dists_expanded + dist_expanded  # (len(E), P)
                
                # The index tensor must have the same number of dimensions as the source.
                # We expand hw_tensor to match the shape of lb_batch and ub_batch.
                index_expanded = hw_tensor.unsqueeze(1).expand_as(lb_batch)

                # Use scatter_reduce_ for in-place, vectorized updates.
                # 'amax' is equivalent to a scattered maximum operation.
                # 'include_self=True' ensures the update is torch.maximum(current_value, new_value).
                feature_lower_bound_distances.scatter_reduce_(
                    0, index_expanded, lb_batch, reduce='amax', include_self=True
                )
                # 'amin' is equivalent to a scattered minimum operation.
                feature_upper_bound_distances.scatter_reduce_(
                    0, index_expanded, ub_batch, reduce='amin', include_self=True
)

        elif paradigm == "hypersphere":
            # Use pre-computed total_hw and reuse estimated centers/radii
            estimated_centers = torch.zeros((self.total_hw, self.D), device=self.device)
            estimated_radii = torch.ones((self.total_hw, 1), device=self.device) * float("inf")
            
            
            # Track which feature vectors have been updated beyond initialization
            updated_feature_vectors = set()
            
            for (hw, p), v in E.items():  # remains in order of elements added (i.e. for the same hw, if multiple p's exist, they are processed in the order they were added)
                feature_lower_bound_distances[hw, p] = v
                feature_upper_bound_distances[hw, p] = v
                
                if hw not in updated_feature_vectors:
                    # Place center at prototype position with distance as radius
                    estimated_centers[hw] = self.prototypes[p].squeeze()
                    estimated_radii[hw] = v
                    updated_feature_vectors.add(hw)
                else:
                    # Refine the estimated sphere by intersecting with the new known sphere
                    c1 = self.prototypes[p].squeeze() # New known prototype center
                    c2 = estimated_centers[hw]        # Current estimated feature center
                    r1 = v                     # New known radius (distance to new proto)
                    r2 = estimated_radii[hw]          # Current estimated radius

                    
                    assert r2.shape == (1,), f"Estimated radius shape: {r2.shape}"
                    assert r2 > 0, "Negative radius or zero radius"
                    # print(f"r2 : {type(r2)}")
                    # c1_c2 = torch.norm(c1 - c2)  # (1)
                    # c1_c2 = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0)) ** 2  # (1)
                    # d_squared = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0)).pow(2).squeeze() # Ensure scalar d^2
                    d_squared = torch.norm(c1[None] - c2[None], dim=-1).pow(2).squeeze()  # (1) squared distance
                    # d_alt_squared = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0)).pow(2).squeeze()  # (1) squared distance
                    # assert torch.isclose(d_squared, d_alt_squared, atol=self.epsilon), f"Distance mismatch: {d_squared} vs {d_alt_squared}"
                    assert d_squared >= 0, f"Negative squared distance: {d_squared}"
                    
                    # Add small epsilon for stability if d can be zero, although assertion should prevent this
                    d = torch.sqrt(d_squared + self.epsilon)  # Calculate d
                    # h = 1 / ((2 * c1_c2) * (r1**2 - r2**2 + c1_c2**2))  # (1)
                    r1_val = r1.item() if torch.is_tensor(r1) else r1  # Ensure scalar
                    r2_val = r2.item() if torch.is_tensor(r2) else r2  # Ensure scalar

                    h = (r1_val**2 - r2_val**2 + d_squared) / (2 * d + self.epsilon) # Add epsilon to denominator for stability
                    h_inv = (r2_val**2 - r1_val**2 + d_squared) / (2 * d + self.epsilon)  # (1)
                    
                    sqrt_argument = r1**2 - h**2
                    # Clamp negative values resulting from float errors to 0 before sqrt
                    r3 = torch.sqrt(torch.relu(sqrt_argument))
                    r3_val = r3.item() if torch.is_tensor(r3) else r3
                    if torch.any(sqrt_argument < 0):
                        print(f"Warning: Clamped negative sqrt argument for r3 at hw={hw}")
                    # print(f"r3 : {type(r3)}")
                    unit_vector_c1_c2 = (c2 - c1) / (d + self.epsilon) # Add epsilon
                    unit_vector_c2_c1 = (c1 - c2) / (d + self.epsilon)  # (D)
                    c3 = c1 + h * unit_vector_c1_c2
                    c3_alt = c2 + h_inv * unit_vector_c2_c1  # (D)
                    if not (torch.isclose(c3, c3_alt, atol=self.epsilon).all(), f"Centers mismatch: distance {torch.norm(c3 - c3_alt)}"):
                        print(f"Warning: Centers mismatch at hw={hw}, c3={c3}, c3_alt={c3_alt}")
                        warning_counter += 1
                    assert c3.shape == (self.D,), f"Estimated center shape: {c3.shape}"

                    if estimated_radii[hw] < r3:
                        # if warning_counter < 1:
                        #     print(f"WARNING! previous r3:{estimated_radii[hw].item()}, new r3:{r3.item()}")
                        warning_counter += 1
                        
                    # estimated_centers[hw] = c3  # (D)
                    estimated_centers[hw] = c3_alt  # (D)
                    estimated_radii[hw] = r3  # (1)
                    
            # Compute center-to-prototype distances for all prototypes at once
            proto_positions = self.prototypes.squeeze()  # (P, D)
            if len(proto_positions.shape) == 1:
                proto_positions = proto_positions.unsqueeze(0)
            
            # Broadcast computation: (H*W, 1, D) - (1, P, D) -> (H*W, P, D)
            center_to_proto_diffs = estimated_centers.unsqueeze(1) - proto_positions.unsqueeze(0)
            center_to_proto_dists = torch.norm(center_to_proto_diffs, dim=2)  # (H*W, P)
            
            # Update all bounds at once
            radii_broadcasted = estimated_radii.squeeze(1).unsqueeze(1)  # (H*W, 1)
            
            new_lower_bounds = center_to_proto_dists - radii_broadcasted  # (H*W, P)
            new_upper_bounds = center_to_proto_dists + radii_broadcasted  # (H*W, P)
            
            feature_lower_bound_distances = torch.maximum(feature_lower_bound_distances, new_lower_bounds)
            feature_upper_bound_distances = torch.minimum(feature_upper_bound_distances, new_upper_bounds)
            
            # for the `next_pairs` method we need the estimated centers and radii
            self.estimated_centers = estimated_centers
            self.estimated_radii = estimated_radii
            
        else:
            raise ValueError(f"Unknown paradigm: {paradigm}. Supported paradigms are 'triangle' and 'hypersphere'.")
        
        # update global bounds
        lower_bound_distances = torch.min(feature_lower_bound_distances, dim=0).values  # (H*W, P) -> (P)
        upper_bound_distances = torch.min(feature_upper_bound_distances, dim=0).values  # (H*W, P) -> (P)
        
        return (lower_bound_distances, upper_bound_distances)
    
    def _verify_explanation(self, lower_bound, upper_bound, unverified_classes):
        # verify conditions
        weights = self.weights  # (P, K) # last layer of the ProtoPNet architecture
        lower_bound_sim = self.model.classifier.similarity_layer.distances_to_similarities(
            upper_bound
        )  # biggest distance -> smallest similarity
        upper_bound_sim = self.model.classifier.similarity_layer.distances_to_similarities(
            lower_bound
        )  # smallest distance -> biggest similarity
        # treat nans as zero
        lower_bound_sim = torch.nan_to_num(lower_bound_sim, 0.0)
        # lower_bound_sim *= torch.logical_not(lower_bound_sim.isnan) + 0*torch.isnan(lower_bound_sim)
        # lower_bound, upper_bound # (N, P)
        with torch.no_grad():
            selected_class = self.c  # (1) # predicted class
            predicted_class_weights = weights[selected_class]  # (P)
            batch_selector = weights > predicted_class_weights  # (P, K) # boolean matrix
            similarities_to_check = upper_bound_sim * batch_selector + lower_bound_sim * (
                torch.logical_not(batch_selector)
            )  # (P, K)
            # res(i) = upper_bound_sim(i) if w_{i,k} > w_{i,c} else lower_bound_sim(i)
            # similarities_to_check = similarities_to_check[:, unverified] # only classes that were not verified beforehand # actually NO
            # check = similarities_to_check.swapaxes(0,1) # (K, P)
            decision_output = self.decision_head(similarities_to_check)  # (K, K)
            
        
        # print(decision_output.shape)
        new_unverified = unverified_classes.copy()  # Copy the list of unverified classes
        # unverified_conf = torch.zeros(self.num_classes)
        unverified_conf = torch.ones(self.num_classes) * -1 * float("inf")  # Initialize with negative infinity for unverified classes
        
        for uidx in unverified_classes:
            # print(f"Best counterfactual for class {uidx}: {decision_output[uidx, uidx]:.3f} vs {decision_output[uidx, selected_class]:.3f}")
            if decision_output[uidx, selected_class] > decision_output[uidx, uidx]:
                new_unverified.remove(uidx)
            else:
                unverified_conf[uidx] = decision_output[uidx, uidx]
        unverified = new_unverified
        
        return unverified, torch.tensor(unverified_conf, device=self.device).max(dim=0)[0]  # Return the maximum confidence of the unverified classes
    
    ###############################
    # Next pair selection methods #
    ###############################

    def _next_pair_round_robin(self, E):
        """
        Simplest approach: Round-robin through feature vectors, but order them by potential impact.
        """
        if self.paradigm == "hypersphere":
            # Count selections per feature vector
            feature_counts = torch.zeros(self.total_hw, device=self.device)
            for (hw, p) in E.keys():
                feature_counts[hw] += 1
            
            # Find feature vectors with minimum selections
            min_selections = torch.min(feature_counts).item()
            candidate_features = torch.where(feature_counts == min_selections)[0]
            
            # Among candidates, pick the one with highest potential cosine similarity
            best_hw = None
            best_p = None
            best_score = -1
            
            for hw in candidate_features:
                hw_val = hw.item()
                for p in range(self.num_prototypes):
                    if (hw_val, p) not in E:  # Not already selected
                        # Compute cosine similarity for this pair
                        center = self.estimated_centers[hw_val]
                        proto_pos = self.prototypes[p].squeeze()
                        feat_pos = self.z.flatten(start_dim=2).swapaxes(1,2).swapaxes(0,1)[hw_val, 0]
                        
                        center_to_proto = proto_pos - center
                        center_to_feat = feat_pos - center
                        
                        cos_sim = torch.abs(torch.nn.functional.cosine_similarity(
                            center_to_proto.unsqueeze(0), 
                            center_to_feat.unsqueeze(0), 
                            dim=1
                        )).item()
                        
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best_hw = hw_val
                            best_p = p
            
            return (best_hw, best_p) if best_hw is not None else (0, 0)
        
    def _next_pair_round_robin_vectorized(self, E):
        """
        A vectorized and optimized version of the round-robin selection strategy.
        """
        if self.paradigm == "hypersphere":
            # 1. VECTORIZED COUNTING
            # If the explanation is empty, all features are candidates with 0 selections.
            if not E:
                feature_counts = torch.zeros(self.total_hw, dtype=torch.long, device=self.device)
            else:
                # Get all feature vector indices from the explanation keys.
                hw_indices = torch.tensor([hw for hw, p in E.keys()], dtype=torch.long, device=self.device)
                # Count occurrences of each feature vector index in a single operation.
                feature_counts = torch.bincount(hw_indices, minlength=self.total_hw)

            # Find the set of feature vectors that have been selected the minimum number of times.
            min_selections = torch.min(feature_counts)
            candidate_features = torch.where(feature_counts == min_selections)[0]
            
            if candidate_features.numel() == 0:
                # Edge case: if no candidates are found, fallback to a default.
                return 0, 0

            # 2. BATCH COMPUTATION OF SCORES
            # Pre-fetch all data needed for the candidate features.
            candidate_centers = self.estimated_centers[candidate_features]      # Shape: (num_candidates, D)
            all_protos = self.prototypes.squeeze()                              # Shape: (P, D)
            
            # Pre-calculate feature positions ONCE.
            feat_pos_all = self.z.flatten(start_dim=2).permute(2, 0, 1).squeeze(1) # Shape: (H*W, D)
            candidate_feat_pos = feat_pos_all[candidate_features]               # Shape: (num_candidates, D)

            # Compute all vectors needed for cosine similarity using broadcasting.
            # `centers_to_protos` shape: (num_candidates, P, D)
            centers_to_protos = all_protos.unsqueeze(0) - candidate_centers.unsqueeze(1)
            # `centers_to_feats` shape: (num_candidates, 1, D)
            centers_to_feats = candidate_feat_pos.unsqueeze(1) - candidate_centers.unsqueeze(1)
            
            # Compute all cosine similarities at once.
            # Resulting shape: (num_candidates, P)
            scores = torch.abs(torch.nn.functional.cosine_similarity(centers_to_protos, centers_to_feats, dim=2))

            # 3. VECTORIZED MASKING
            # Create a boolean mask to filter out pairs already present in the explanation E.
            mask = torch.ones_like(scores, dtype=torch.bool)
            if E:
                # Build a sparse representation of E for efficient lookup
                e_keys_tensor = torch.tensor(list(E.keys()), dtype=torch.long, device=self.device)
                # For each candidate, find which prototypes are already in E
                for i, hw in enumerate(candidate_features):
                    # Find all pairs in E that match the current candidate feature vector
                    p_indices_in_e = e_keys_tensor[e_keys_tensor[:, 0] == hw, 1]
                    if p_indices_in_e.numel() > 0:
                        mask[i, p_indices_in_e] = False

            # Apply the mask. Invalid pairs get a score of -1 so they won't be picked.
            scores[~mask] = -1.0

            # 4. VECTORIZED SEARCH
            # Find the index of the highest score in the flattened score tensor.
            if scores.numel() == 0 or torch.all(scores == -1.0):
                # Fallback if all possible pairs for candidates are already selected
                return 0, 0 
                
            flat_idx = torch.argmax(scores.flatten())
            
            # Convert the flat index back to 2D indices: (candidate_idx, prototype_idx).
            candidate_idx = flat_idx // self.num_prototypes
            best_p = (flat_idx % self.num_prototypes).item()
            
            # Retrieve the original feature vector index from the candidates tensor.
            best_hw = candidate_features[candidate_idx].item()
            
            return best_hw, best_p
        
        elif self.paradigm == "triangle":
            if not E:
                feature_counts = torch.zeros(self.total_hw, dtype=torch.long, device=self.device)
            else:
                hw_indices = torch.tensor([hw for hw, p in E.keys()], dtype=torch.long, device=self.device)
                feature_counts = torch.bincount(hw_indices, minlength=self.total_hw)

            min_selections = torch.min(feature_counts)
            candidate_features = torch.where(feature_counts == min_selections)[0]

            if candidate_features.numel() == 0:
                return 0, 0

            # Get the distances for candidate features
            candidate_distances = self._distances_transposed[candidate_features, :]

            mask = torch.ones_like(candidate_distances, dtype=torch.bool)
            if E:
                e_keys_tensor = torch.tensor(list(E.keys()), dtype=torch.long, device=self.device)
                for i, hw in enumerate(candidate_features):
                    p_indices_in_e = e_keys_tensor[e_keys_tensor[:, 0] == hw, 1]
                    if p_indices_in_e.numel() > 0:
                        mask[i, p_indices_in_e] = False

            candidate_distances[~mask] = float('inf')

            flat_idx = torch.argmin(candidate_distances.flatten())
            candidate_idx = flat_idx // self.num_prototypes
            best_p = (flat_idx % self.num_prototypes).item()
            best_hw = candidate_features[candidate_idx].item()

            return best_hw, best_p

        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}. Supported paradigms are 'triangle' and 'hypersphere'.")

    def _next_pair_progressive(self, E):
        """
        Alternative approach: Progressive selection strategy that naturally avoids repetitive selections.
        """
        if self.paradigm == "hypersphere":
            # Group existing pairs by feature vector
            feature_to_prototypes = {}
            for (hw, p) in E.keys():
                if hw not in feature_to_prototypes:
                    feature_to_prototypes[hw] = []
                feature_to_prototypes[hw].append(p)
            
            # Create mask for existing pairs
            mask = torch.ones((self.total_hw, self.num_prototypes), dtype=torch.bool, device=self.device)
            if E:
                existing_hw, existing_p = zip(*E.keys())
                existing_hw_tensor = torch.tensor(existing_hw, device=self.device)
                existing_p_tensor = torch.tensor(existing_p, device=self.device)
                mask[existing_hw_tensor, existing_p_tensor] = False
            
            # Compute base scores
            centers_to_protos = self.prototypes.squeeze((2,3)).squeeze(0) - self.estimated_centers.unsqueeze(1) 
            centers_to_feats = self.z.flatten(start_dim=2).swapaxes(1,2).swapaxes(0,1) - self.estimated_centers.unsqueeze(1)
            cos = nn.CosineSimilarity(div=2, eps=1e-8)
            cos_sim = cos(centers_to_protos, centers_to_feats)
            base_scores = torch.abs(cos_sim)
            
            # PROGRESSIVE PENALTY: Apply exponentially increasing penalty for repeated feature selections
            penalty_factor = torch.ones_like(base_scores)
            for hw in range(self.total_hw):
                if hw in feature_to_prototypes:
                    num_selections = len(feature_to_prototypes[hw])
                    # Exponential decay: each additional selection gets exponentially less likely
                    penalty_factor[hw, :] *= (0.5 ** num_selections)
            
            # Apply progressive penalty
            adjusted_scores = base_scores * penalty_factor
            adjusted_scores[~mask] = 0.0
            
            max_idx = torch.argmax(adjusted_scores)
            best_pair = (max_idx // self.num_prototypes, max_idx % self.num_prototypes)
            
            return best_pair

###############################################################################

###############################################################################


def main(top_k: bool = True, triangle: bool = False, hypersphere: bool = False):
    # print("Inspecting the class...")
    # _inspect_class(FormalExplanationBase)
    # sys.exit()
    print("Starting the explanation process...")
    # load the model

    seed = 1  # 1338

    # print("Loading configuration...")
    # Current working directory
    cwd = Path(os.getcwd())
    # print("Current working directory:", cwd)
    # Path to the trained model
    # final_model_path = cwd / "logs" / f"protopnet_cub200_{seed}" / "final" 
    final_model_path = cwd / "logs" / "vgg_1" / "final"
    prototype_path = final_model_path / "prototypes.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model, test_loader = load_model(final_model_path, seed=seed, device=device, test_set=True)
    # test_loader = dataloaders["test_set"]

    # get a sample image
    batch_size = 1  # 16
    test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=batch_size, shuffle=False)
    
    save_path = cwd / "images"
    # verif_counter = 0
    print("Starting verification...")
    print("Number of samples: ", len(test_loader))
    MAX_PROTOTYPES = 2_000
    data = []
    correct_data = []
    incorrect_data = []
    interesting_data = []
    # MAX_ITER = -1  # set to -1 to run on all samples, 0 to ignore
    MAX_ITER = 1
    
    for i, (x, y) in enumerate(test_loader):
        if i >= MAX_ITER > 0:
            break
        x = x.to(device)
        y = y.to(device)
        # print(f"Input shape: {x.shape}, Label shape: {y.shape}")
        # print(f"Input: {x}, Label: {y}")
        # print(f"Input: {x[0,0,0,0]}, Label: {y[0]}")
        # print(f"Input: {x[0,0,0,:]}, Label: {y[0]}")
        
        if top_k:
            explanation = TopKFormalExplanation(model, device=device)
            explanation.explain_one(x, y)
        elif triangle:
            explanation = SpatialFormalExplanation(model, device=device, paradigm="triangle", save_proto=True, load_proto=False, prototype_filepath=prototype_path)
            explanation.explain_one(x, y, verbose=True)
        elif hypersphere:
            explanation = SpatialFormalExplanation(model, device=device, paradigm="hypersphere", save_proto=True, load_proto=False, prototype_filepath=prototype_path)
            explanation.explain_one(x, y, verbose=True)
        else:
            raise ValueError("At least one of top_k, triangle, or hypersphere must be True.")

        data.append(explanation.explanation)
        if explanation.c == y.item():
            correct_data.append(explanation.explanation)
        else:
            incorrect_data.append(explanation.explanation)


if __name__ == "__main__":
    print("Entering main...")
    # load the model
    args = parse_args()
    main(args.top_k, args.triangle, args.hypersphere)
    # main(0, 0, 1)  # For testing purposes, run with triangle inequality only