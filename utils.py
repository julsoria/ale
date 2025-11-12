import sys
from pathlib import Path
import psutil
import torch
from typing import Callable
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.data import DatasetManager as DM


def check_memory_usage(threshold_mb=500):
    """Check current memory usage and raise an error if it exceeds the threshold."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB

    if memory_used_mb > threshold_mb:
        print(f"Memory usage exceeded! Current usage: {memory_used_mb:.2f} MB")
        sys.exit("Terminating script due to high memory usage.")


# --- Wrapper Function ---
def _create_sqrt_wrapper(original_distance_func: Callable, eps: float = 1e-12) -> Callable:
    """Creates a wrapper that applies sqrt to the output of a distance function."""
    def sqrt_distance_wrapper(*args, **kwargs):
        # Calculate original squared distance
        dist_sq = original_distance_func(*args, **kwargs)
        # Clamp slightly above zero before sqrt for numerical stability
        # (prevents NaN gradients if ever used, and handles potential tiny negatives)
        dist = torch.sqrt(torch.clamp(dist_sq, min=eps))
        # dist = torch.sqrt(dist_sq)
        return dist
    return sqrt_distance_wrapper


def load_model(final_model_path: Path, seed:int, device: str = "cuda:0", test_set: bool = True):
    model_path = final_model_path / "model_state.pth"
    model_config_path = final_model_path / "model_arch.yml"
    data_config_path = final_model_path / "dataset.yml"
    # Load the model
    print("Loading the model...")
    print("Model config path: ", model_config_path)
    model = CaBRNet.build_from_config(config=str(model_config_path), state_dict_path=model_path, seed=seed)
    # Print the model
    # print(model.__class__)
    # Load the test data
    print("Loading the test data...")
    dataloaders = DM.get_dataloaders(config=str(data_config_path))
    # print(dataloaders.keys())
    # print('Evaluating the model...')
    # res = model.evaluate(dataloaders['test_set'], device='cuda:0', verbose=True)
    # print('Results:', res)
    test_loader = dataloaders["test_set"]
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print("Device: ", device)
    model.to(device)
    if test_set:
        model.eval()
        return model, test_loader
    else:
        return model
