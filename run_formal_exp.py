import argparse
import os
from pathlib import Path
import torch
import logging
import psutil
from utils import load_model, check_memory_usage
from ale_methods import FormalExplanationBase, TopKFormalExplanation, SpatialFormalExplanation
from tqdm import tqdm, trange
import time
CWD = Path(os.getcwd())


ACCEPTED_PARADIGMS = ["triangle", "hypersphere", "top_k"]
SPATIAL_PARADIGMS = ["hypersphere", "triangle"]
# ACCEPTED_DATASETS = ["stanford_cars", "cub200", "flowers102", "oxford_iiit_pet", "cifar10", "cifar100", "mnist"]
ACCEPTED_DATASETS = {"stanford_cars": 196, "cub200": 200, "flowers102": 102, "oxford_iiit_pet": 37, "cifar10": 10, "cifar100": 100, "mnist": 10}
ACCEPTED_ARCHITECTURES = ["protopnet", "protopool"]

logger = logging.getLogger(__name__)


def get_random_idx(dataloader, num_per_class=5, flatten=True):
    """
    Get random indices for each class in the dataloader.
    """
    # Get the number of classes from the dataloader
    if 'classes' in dataloader.dataset.__dict__:
        num_classes = len(dataloader.dataset.classes)
    elif '_labels' in dataloader.dataset.__dict__:
        num_classes = max(dataloader.dataset._labels) + 1
    elif 'targets' in dataloader.dataset.__dict__:
        num_classes = max(dataloader.dataset.targets) + 1
    else:
        raise ValueError("Dataset does not have 'classes', '_labels', or 'targets' attribute to determine number of classes.")
    # Create a list to store the indices for each class
    idx_per_class = [[] for _ in range(num_classes)]
    # Iterate through the dataloader
    for i, (_, label) in enumerate(dataloader):
        # Get the label for the current batch
        label = label.item()
        # Append the index to the corresponding class list
        idx_per_class[label].append(i)
        # If the class list has reached the desired number of indices, break
        if len(idx_per_class[label]) >= num_per_class:
            continue
        
    #  shuffle the indices for each class
    for class_idx in trange(num_classes):
        torch.manual_seed(0)
        class_indices = torch.tensor(idx_per_class[class_idx])
        # print(f"Class {class_idx} indices: {class_indices}")
        indices = torch.randperm(len(class_indices))
        indices = indices[:num_per_class]
        # print(f"Random indices for class {class_idx}: {class_indices[indices]}")
        idx_per_class[class_idx] = class_indices[indices].tolist()
    # Remove the empty lists from the list of indices
    if flatten:
        # Flatten the list of indices
        idx_per_class = [idx for class_indices in idx_per_class for idx in class_indices]
    else:
        pass
    # Return the indices for each class
    return idx_per_class


def get_latest_iteration(folder_path):
    """
    Get the latest image explained from the folder.
    Format: folder_path/<iteration_number>.txt
    """
    # Get a list of all files in the directory
    files = os.listdir(folder_path)
    # Filter the list to only include files that match the pattern <iteration_number>.txt
    iteration_files = [f for f in files if f.endswith(".txt") and f[:-4].isdigit()]
    # Extract the iteration numbers from the filenames and convert them to integers
    iteration_numbers = [int(f[:-4]) for f in iteration_files]
    # Return the maximum iteration number
    return max(iteration_numbers) if iteration_numbers else None


def get_all_iterations(folder_path):
    """
    Get all the iterations from the folder.
    Format: folder_path/<iteration_number>.txt
    """
    # Get a list of all files in the directory
    files = os.listdir(folder_path)
    # Filter the list to only include files that match the pattern <iteration_number>.txt
    iteration_files = [f for f in files if f.endswith(".txt") and f[:-4].isdigit()]
    # Extract the iteration numbers from the filenames and convert them to integers
    iteration_numbers = [int(f[:-4]) for f in iteration_files]
    # Return the maximum iteration number
    return iteration_numbers if iteration_numbers else None


def exp_check(formal_explanation, model, test_loader, explanation_path, device):
    img, label = next(iter(test_loader))
    iter_idx = 0
    img = img.to(device)
    label = label.to(device)
    formal_explanation.explain(
        img,
        label,
        top_k=True,
        max_only=False,
        triangle_inequality=False,  # TRIANGLE_INEQUALITY,
        hypersphere_approximation=False,  # HYPERSPHERE_APPROXIMATION,
        verbose=False,
    )
    explanations = formal_explanation.explanation

    # Print the explanations
    print("Generated Explanations:")
    # for explanation in explanations:
    #     print(explanation)
    print(list([el[0] for el in explanations]))
    print(f"Explanation length: {len(explanations)}")
    predicted_label = int(torch.argmax(model(img)[0], dim=1))
    print(f"Predicted label: {predicted_label}")

    explanation_folder = explanation_path / "top_k_explanations"
    with open(explanation_folder / f"{iter_idx}.txt", "r") as f:
        for line in f:
            print(line.strip())


def run_indices(explainer, model, test_loader, explanation_path, device, indices, paradigm: str = "triangle", overwrite: bool = False) -> int:
    """
    Run the explanation for the given indices.
    """
    model.eval()
    # Explainer must be an instance of FormalExplanationBase or its subclass.
    assert isinstance(explainer, FormalExplanationBase), "Explainer must be an instance of FormalExplanationBase or its subclass."
    # Explainer must have a method 'explain'.
    assert hasattr(explainer, 'explain'), "Explainer must have a method 'explain'."
    # Paradigm must be one of the accepted paradigms.
    assert paradigm in ["top_k", "triangle", "hypersphere"], "Paradigm is not valid"

    if paradigm == "top_k":
        paradigm_txt = "Top-k"
        # run the experiment for all test samples
        # indices = list(range(len(test_loader)))
    elif paradigm == "triangle":
        paradigm_txt = "Triangle inequality"
    elif paradigm == "hypersphere":
        paradigm_txt = "Hypersphere approximation"

    n_samples = 0
    for idx, (img, label) in tqdm(enumerate(test_loader), total=len(test_loader)):
        check_memory_usage(threshold_mb=5000)
        if idx not in indices:
            # print(f"Index {idx} not in the list of indices.")
            continue
        # Check if the index exists in the file
        idx_file = explanation_path / f"{paradigm}_explanations" / f"{idx}.txt"
        # print(f"Index file: {idx_file}")
        # Do not check if the file exists for time benchmark
        if idx_file.exists() and not overwrite:
            with open(idx_file, "r") as f:
                lines = f.readlines()
                if len(lines) > 0:
                    print(f"Index {idx} already exists in the file.")
                    continue
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            torch.cuda.empty_cache()
        # output = model(img)
        # try:
        #     predicted_label = int(torch.argmax(output[0], dim=1))
        # except ValueError as e:
        #     print(f"Value error: {e}")
        #     print(f"Output shape: {output[0].shape}")
        #     print(f"Image shape: {img.shape}")
        #     print(f"Label shape: {label.shape}")
        #     continue
        try:
            explainer.explain(
                img,
                label,
                verbose=True,
            )
            n_samples += 1
            explanations = explainer.explanation
            exp_size = explainer.explanations_size
            correct_explanations = explainer.correct_explanations
            incorrect_explanations = explainer.incorrect_explanations
        except AssertionError as e:
            print(f"Assertion error: {e}")
            explanations = []
            exp_size = correct_explanations = incorrect_explanations = []
        except ValueError as e:
            print(f"Value error: {e}")
            explanations = []
            exp_size = correct_explanations = incorrect_explanations = []
        
        with open(idx_file, "w") as f:
            if isinstance(explainer.explanations_size, list):
                f.write(f"{paradigm_txt} explanation size: ")
                f.write(str(exp_size))
                f.write("\n")
            if explainer.correct_explanations != []:
                f.write(f"{paradigm_txt} correct explanations: ")
                f.write(str(correct_explanations))
                f.write("\n")
            if explainer.incorrect_explanations != []:
                f.write(f"{paradigm_txt} incorrect explanations: ")
                f.write(str(incorrect_explanations))
                f.write("\n")
            if explainer.explanations_size != []:
                f.write(f"{paradigm_txt} explanation: ")
                f.write(str(explanations))
                f.write("\n")
                
    return n_samples  # Return the number of samples processed


def check_indices(indices, path):
    """
    For all indices, check if the file exists in the path.
    Return True if all files exist, False otherwise.
    """
    # create the path and parent directories if they do not exist
    if not os.path.exists(path):
        os.makedirs(path)
    files = os.listdir(path)
    for idx in indices:
        file_name = f"{idx}.txt"
        if file_name not in files:
            print(f"File {file_name} does not exist.")
            return False
    return True


def main(paradigm="triangle", overwrite=False, all_indices=False, seed=42, data="stanford_cars", arch="protopnet") -> int:
    """
    Main function to run the explanation.
    """
    # print("Current working directory:", CWD)
    # Path to the trained model
    # final_model_path = CWD / "logs" / f"protopnet_cub200_{seed}" / "final"
    # data = "cub200"  # or "cub200"
    if data not in ACCEPTED_DATASETS:
        raise ValueError(f"Data {data} is not valid. Options are: {ACCEPTED_DATASETS.keys()}")
    if arch not in ACCEPTED_ARCHITECTURES:
        raise ValueError(f"Architecture {arch} is not valid. Options are: {ACCEPTED_ARCHITECTURES}")

    final_model_path = CWD / 'logs' / f'{arch}_{data}_{seed}' / 'final'
    print(f"Final model path: {final_model_path}")
    if not final_model_path.exists():
        raise FileNotFoundError(f"Final model path {final_model_path} does not exist. Please check the path or the seed.")
            
    prototype_path = final_model_path / "prototypes.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    explanation_path = final_model_path / "explanations"
    print(explanation_path)
    # explanation_path = CWD / "images" / "vgg_1" / "explanations"
    if not explanation_path.exists():
        explanation_path.mkdir(parents=True, exist_ok=True)

    model, test_loader = load_model(final_model_path, seed=seed, device=device, test_set=True)
    # test_loader = dataloaders["test_set"]
    if test_loader.batch_size != 1:
        test_loader = torch.utils.data.DataLoader(test_loader.dataset, batch_size=1, shuffle=False)

    if paradigm in SPATIAL_PARADIGMS:
        formal_explanation: FormalExplanationBase = SpatialFormalExplanation(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False, paradigm=paradigm)
    elif paradigm == "top_k":
        formal_explanation: FormalExplanationBase = TopKFormalExplanation(model, device, prototype_filepath=prototype_path, save_proto=True, load_proto=False)
    # iters = get_all_iterations(explanation_path/ "triangle_explanations")
    # print(f"Iterations: {len(iters)}")
    if all_indices:
        random_idx = list(range(len(test_loader)))
        print(f"Running explanation for all indices in the test set: {len(random_idx)}")
    else:
        random_idx_file = final_model_path / "random_indices.txt"
        # random_idx_file = CWD / "random_indices.txt"
        if random_idx_file.exists():
            with open(random_idx_file, "r") as f:
                random_idx = [int(line.strip()) for line in f.readlines()]
            print(f"Loaded {len(random_idx)} random indices from file.")
        else:
            print(f"File {random_idx_file} does not exist. Generating random indices.")
            # Generate random indices
            numel = 5
            numclasses = ACCEPTED_DATASETS[data]

            random_idx = get_random_idx(test_loader, num_per_class=numel)
            
            assert len(random_idx) == numel*numclasses, f"Expected 5 indices, got {len(random_idx)}"
            print(f"Random indices: {random_idx[:10]}")
            # save the random indices to a file
            with open(random_idx_file, "w") as f:
                for iidx in random_idx:
                    f.write(f"{iidx}\n")
            
            print(f"Saved {len(random_idx)} random indices to file {random_idx_file}.")
    
    if (not check_indices(random_idx, explanation_path / f"{paradigm}_explanations")) or overwrite:
        print(f"Files do not exist for paradigm {paradigm}.")
        n_samples = run_indices(formal_explanation, model, test_loader, explanation_path, device, random_idx, paradigm=paradigm, overwrite=overwrite)
    else:
        print(f"Files exist for paradigm {paradigm}.")
        return 0
    return n_samples


def measure_accuracy(model, testloader, indices):
    total = 0
    correct = 0
    model.eval()
    device = next(model.parameters()).device
    
    # Assuming testloader is a DataLoader object
    for i, (img, label) in enumerate(testloader):
        if i not in indices:
            continue
        # Forward pass
        img = img.to(device)
        label = label.to(device)
        output = model(img)[0]
        predicted = torch.argmax(output, 1)
        # Calculate accuracy
        # Assuming label is a tensor of the same size as predicted
        total += label.size(0)
        correct += (predicted == label).sum().item()
    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Run formal explanation.")
    argparser.add_argument(
        "--paradigm",
        type=str,
        default="top_k",
        help="Default=top_k. Paradigm to use for explanation. Options: triangle, hypersphere, top_k",
        choices=["triangle", "hypersphere", "top_k"]
    )
    argparser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing explanations (if they exist).",
    )
    argparser.add_argument(
        "--all_indices",
        action="store_true",
        help="Run the explanation for all indices in the test set.",
    )
    argparser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default=42.",
        # required=True
    )
    argparser.add_argument(
        "-d",
        "--dataset",
        # "--data",
        type=str,
        default="stanford_cars",
        help="Dataset to use for explanation. Default=stanford_cars.",
        choices=ACCEPTED_DATASETS.keys()
    )
    argparser.add_argument(
        "-m",
        "--model-arch",
        # "--arch",
        type=str,
        default="protopnet",
        help="Model architecture to use for explanation. Default=protopnet",
        choices=ACCEPTED_ARCHITECTURES
    )
    args = argparser.parse_args()
    print(args)
    paradigm = args.paradigm
    assert paradigm in ["triangle", "hypersphere", "top_k"], f"Paradigm {paradigm} is not valid"
    start_time = time.time()
    n_samples = main(paradigm=paradigm, overwrite=args.overwrite, all_indices=args.all_indices, seed=args.seed, data=args.dataset, arch=args.model_arch)
    end_time = time.time()
    total_time = end_time - start_time
    total_memory = psutil.Process().memory_info().rss / (1024 * 1024)  # Convert bytes to MB
    print(f"Total memory used: {total_memory:.2f} MB")
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Total time taken: {total_time / 60:.4f} minutes ({total_time / 3600:.4f} hours)")
    print(f"Average time per sample: {total_time / n_samples:.4f} seconds" if n_samples > 0 else "No samples processed.")
    if n_samples > 0:
        print(f"Number of samples processed: {n_samples}")
    print("Formal explanation completed.")
