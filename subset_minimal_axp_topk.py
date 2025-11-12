import torch

from tqdm import tqdm

from subset_minimal_axp_base import FormalExplanationBase
from utils import check_memory_usage


###############################################################################

class TopKFormalExplanation(FormalExplanationBase):
    """
    A class to generate formal explanations for the prediction of a model using the top-k prototypes.
    Inherits from FormalExplanationBase.
    """

    def __init__(self, model, device="cuda:0", **kwargs):
        super().__init__(model, device=device, **kwargs)
        self.explanation = []  # List to store the explanation
        
    def explain_one(self, x, y, verbose=False):
        """
        Generate a formal explanation for a single input x with label y.
        Args:
            x: Input tensor.
            y: True label of the input.
            verbose: If True, print additional information.
        """
        # self.forward(x, y, verbose=verbose, top_k=True, max_only=False, triangle_inequality=False, hypersphere_approximation=False)
        # exp = self.explanation.copy()  # Copy the explanation to return
        # self.explanation = []  # Clear the explanation for the next call
        # return exp
        
        ### PSEUDO CODE ###.
        # 0. Initialize the explanation E = {} and the unverified classes unverified = {0, ..., C} \ {c}
        # 1. Forward pass
        # 1._. While there are unverified classes:
        # 1.a. Get j = NextPrototype(E) # where j is the most activated prototype not in E
        # 1.c. Update the explanation $E$ with the new prototype j and the associated activation and distance
        # 1.d. Verify the explanation with the activation bounds i.e. check how many unverified classes remain
        # 1.e. If there are unverified classes, go back to step 1
        ### END PSEUDO CODE ###
        
        check_memory_usage(threshold_mb=5000)
        
        x = x.to(self.device)  # (N, C, H, W) # images
        y = y.to(self.device)  # (N) # labels
        self.batch_size = x.size(0)  # N
        
        # Get the model's output
        with torch.no_grad():
            similarities = self.model.similarities(x)  # (N, P, H, W)
            logits = self.model(x)[0]  # (N, K)
            self.batch_size = x.shape[0]  # (N)
        
        prediction_conf, predicted_class = torch.max(logits.squeeze(), dim=0)  # (N, K) -> (N)
        self.c = int(predicted_class.item())  # (1) # predicted class
        
        A = torch.max(similarities[0].view(self.num_prototypes, -1), dim=1).values  # (P)

        E: dict = {}
        
        # verification init
        unverified = list(range(self.num_classes))
        unverified.remove(self.c)
        
        if verbose:
            t = tqdm(
                total=len(unverified),
                desc="Explaining",
                unit="prototype",
                leave=False,
            )
            t.n = 0
            t.refresh()

        while unverified:
            # Get the next prototype
            j, act_j = self._next_prototype(E, A)
            
            # Add the prototype to the explanation
            E.update({j: act_j})  
            
            # update the bounds
            lower_bound, upper_bound = self._update_bounds(E)
            
            unverified, unverified_conf = self._verify_explanation(
                lower_bound, upper_bound, unverified
            )
            
            if verbose:
                t.n += 1
                postfix_str = f"pred cls: {self.c}, true cls: {y.item()}, conf: {prediction_conf:.2f}, next_p: {j:04d}, next_act: {act_j:.3f}, n_unverif: {len(unverified)}, cex conf: {torch.max(unverified_conf):.3f}"
                # postfix_str = f"most-activated proto: {torch}"
                t.set_postfix_str(postfix_str)
                t.refresh()
        
        if verbose:
            t.close()
        self.explanation = []  # empty explanation
        return E  # Return the explanation as a dictionary {prototype_index: activation_value}

    def _next_prototype(self, E, A) -> tuple[int, float]:
        """
        Get the next prototype to add to the explanation.
        Args:
            E: Current explanation.
            A: Activation values of the prototypes.
        Returns:
            j: Index of the next prototype.
            act_j: Activation value of the next prototype.
        """
        # Get the most activated prototype not in E
        existing_p = list(E.keys())
        activations = torch.tensor(A, device=self.device)  # (P)
        activations[existing_p] = float("-inf")  # Set the activations of existing prototypes to -inf
        next_proto_idx: int = int(torch.argmax(activations).item())  # (P)
        next_proto_act = activations[next_proto_idx].item()  # (1)
        return next_proto_idx, next_proto_act

    def _update_bounds(self, E) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the lower and upper bounds of the explanation based on the current explanation E.
        Args:
            E: Current explanation.
        Returns:
            lower_bound: Lower bound of the explanation.
            upper_bound: Upper bound of the explanation.
        """
        lower_bound = torch.zeros(self.num_prototypes, device=self.device)  # (P)
        upper_bound = torch.zeros(self.num_prototypes, device=self.device)  # (P)
        
        lower_bound[list(E.keys())] = torch.tensor(list(E.values()), device=self.device)  # (P)
        upper_bound[list(E.keys())] = torch.tensor(list(E.values()), device=self.device)  # (P)
        
        # Set the upper bound for the prototypes not in E to the minimum activation value in the explanation
        min_activation = torch.min(torch.tensor(list(E.values()), device=self.device))  # (1)
        upper_bound[upper_bound == 0] = min_activation  # (P)
        
        return lower_bound, upper_bound  # (P), (P)

    def _verify_explanation(self, lower_bound, upper_bound, unverified_classes) -> tuple[list, torch.Tensor]:
        # verify conditions
        weights = self.weights  # (P, K) # last layer of the ProtoPNet architecture
        
        with torch.no_grad():
            selected_class = self.c  # (1) # predicted class
            predicted_class_weights = weights[selected_class]  # (P)
            batch_selector = weights > predicted_class_weights  # (P, K) # boolean matrix
            similarities_to_check = upper_bound * batch_selector + lower_bound * (
                torch.logical_not(batch_selector)
            )  # (P, K)
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
    
        

###############################################################################
