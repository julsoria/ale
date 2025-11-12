import torch
import math
from cabrnet.core.utils.similarities import SimilarityLayer


# === Until we have a proper EuclideanDistance class ===
def _euclidean_distances(features: torch.Tensor, prototypes: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""Computes pairwise squared Euclidean (L2) distances between a tensor of features and a tensor of prototypes.

    Args:
        features (tensor): Input tensor. Shape (N, D, H, W).
        prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

    Returns:
        Tensor of distances. Shape (N, P, H, W).
    """
    N, D, H, W = features.shape
    features = features.view((N, D, -1))  # Shape (N, D, HxW)
    features = torch.transpose(features, 1, 2)  # Shape (N, HxW, D)
    prototypes = prototypes.squeeze(dim=(2, 3))  # Shape (P, D)

    # Numerically stable cdist that works on smaller chunks
    def cdist_chunk_prototypes(A, batch_size):
        batch_num = math.ceil(prototypes.size(0) / batch_size)
        return torch.concat(
            [torch.cdist(A, prototypes[i * batch_size : (i + 1) * batch_size]) for i in range(batch_num)], dim=2
        )

    def cdist_chunk_features(batch_size):
        batch_num = math.ceil(N / batch_size)
        return torch.concat(
            [
                cdist_chunk_prototypes(features[i * batch_size : (i + 1) * batch_size], batch_size)
                for i in range(batch_num)
            ]
        )

    distances = cdist_chunk_features(10)
    distances = torch.transpose(distances, 1, 2)  # Shape (N, P, HxW)
    distances = distances.view(distances.shape[:2] + (H, W))  # Shape (N, P, H, W)
    return distances


class EuclideanDistance(SimilarityLayer):
    r"""Layer for computing Euclidean (L2) distances in the convolutional space."""

    def distances(self, features: torch.Tensor, prototypes: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Computes pairwise squared Euclidean (L2) distances between a tensor of features and a tensor of prototypes.

        Args:
            features (tensor): Input tensor. Shape (N, D, H, W).
            prototypes (tensor): Tensor of prototypes. Shape (P, D, 1, 1).

        Returns:
            Tensor of distances. Shape (N, P, H, W).
        """
        N, D, H, W = features.shape
        features = features.view((N, D, -1))  # Shape (N, D, HxW)
        features = torch.transpose(features, 1, 2)  # Shape (N, HxW, D)
        prototypes = prototypes.squeeze(dim=(2, 3))  # Shape (P, D)
        distances = torch.cdist(features, prototypes)  # Shape (N, HxW, P)
        distances = torch.transpose(distances, 1, 2)  # Shape (N, P, HxW)
        distances = distances.view(distances.shape[:2] + (H, W))  # Shape (N, P, H, W)
        return distances


class LogSquaredDistance(SimilarityLayer):
    r"""Abstract layer for computing similarity scores based on the log of distances in the convolutional space.

    Attributes:
        stability_factor: Stability factor.
    """

    def __init__(self, stability_factor: float = 1e-4, **kwargs) -> None:
        r"""Initializes a ProtoPNetDistance layer.

        Args:
            stability_factor (float, optional): Stability factor. Default: 1e-4.
        """
        super().__init__(**kwargs)
        self.stability_factor = stability_factor

    def distances_to_similarities(self, distances: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Converts a tensor of distances into a tensor of similarity scores, such that
        sim=log((1+distances)/(1+epsilon)) where epsilon is the stability factor.

        Args:
            distances (tensor): Input tensor. Any shape.

        Returns:
            Similarity score corresponding to the provided distances. Same shape as input.
        """
        # Ensures that distances are greater than 0
        distances = torch.relu(distances) ** 2
        return torch.log((distances + 1) / (distances + self.stability_factor))


class ProtoPNetSimilarityFormal(EuclideanDistance, LogSquaredDistance):
    r"""Layer for computing similarity scores based on Euclidean (L2) distances in the convolutional space
        (ProtoPNet implementation updated with cdist function).

    Attributes:
        stability_factor: Stability factor.
    """

    def __init__(self, stability_factor: float = 1e-4, **kwargs) -> None:
        r"""Initializes a ProtoPNetSimilarity layer.

        Args:
            stability_factor (float, optional): Stability factor. Default: 1e-4.
        """
        super().__init__(stability_factor=stability_factor, **kwargs)


#######
