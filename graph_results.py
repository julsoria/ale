# Objective:
# We have three result folders, each containing a txt file to parse
# Format: ./images/<paradigm>_explanations/<idx>.txt
# File example: ./images/triangle_explanations/0.txt
# Triangle inequality explanation size: [773]
# Triangle inequality correct explanations: [773]
# Triangle inequality explanation: Formal explanation for class 0:
# [(0, 70, 0.5304415822029114), ... , (l, j, sim(z_l, p_j))]
# We want two things:
# 1. The size of explanations per class
# 2. The size of (in-)correct explanations per class

import collections
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch


def parse_explanations(file_path):
    """
    Parse the explanations from the given file.
    The file is expected to contain lines with the following format:
    "<paradigm> explanation size: [int]"
    "<paradigm> (in-)correct explanations: [int]"
    "<paradigm> explanation: Formal explanation for class <cls_idx>: [... , (l, j, sim(z_l, p_j)), ...]"
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # sizes = []
    # correct_sizes = []
    # incorrect_sizes = []
    size = 0
    correct_size = 0
    incorrect_size = 0

    # if the explanation is empty, we want to skip it
    for line in lines:
        if "explanation size" in line:
            try:
                size = int(line.split(":")[-1].strip().strip("[]"))
                # sizes.append(size)
            except ValueError:
                # Handle the case where the size is not an integer
                # print("Warning: Explanation is empty")
                break
            continue
        elif "incorrect explanations" in line:
            incorrect_size = int(line.split(":")[-1].strip().strip("[]"))
            continue
        elif "correct explanations" in line:
            correct_size = int(line.split(":")[-1].strip().strip("[]"))
            continue
            # correct_sizes.append(correct_size)
            # incorrect_sizes.append(incorrect_size)
        elif "explanation" in line:
            # This line contains the actual explanation
            # We can ignore it for now, as we are only interested in sizes
            pass
        else:
            pass
    return size, correct_size, incorrect_size
    # return sizes, correct_sizes, incorrect_sizes


def parse_paradigm(paradigm, indices, data_folder="./images/", not_all_indices=False) -> tuple:
    """
    Parse the explanations for a given paradigm.
    The paradigm is expected to be one of 'triangle', 'square', or 'circle'.
    """
    # file_path = f"./images/{paradigm}_explanations/"
    file_path = os.path.join(data_folder, f"{paradigm}_explanations/")
    if not os.path.exists(file_path):
        print(f"Warning: The folder {file_path} does not exist.")
        return [], [], []
    
    if (not not_all_indices):
        # we want all files in the folder
        indices = []  # No specific indices for top_k, we want all files
    sizes = []
    correct_sizes = []
    incorrect_sizes = []
    # instead of iterating over all indices in the file, iterate overall indices possible
    for file in os.listdir(file_path):
        if not file.endswith(".txt"):
            continue
        # Extract the index from the file name
        idx = int(file.split(".")[0])
        if idx not in indices and indices != []:
            continue
        # If the index is not in the list, we skip it
        size, correct_size, incorrect_size = parse_explanations(os.path.join(file_path, file))
        sizes.append(size) if size > 0 else 0
        correct_sizes.append(correct_size) if correct_size > 0 else 0
        incorrect_sizes.append(incorrect_size) if incorrect_size > 0 else 0
        # print(f"Parsed {file}: size={size}, correct_size={correct_size}, incorrect_size={incorrect_size}")
    
    # If the sizes are empty, we return empty lists
    if not sizes:
        print(f"Warning: No explanations found for paradigm {paradigm} in {file_path}.")
        return [], [], []

    return sizes, correct_sizes, incorrect_sizes


def plot_explanations(sizes, correct_sizes, incorrect_sizes, paradigm):
    """
    Plot the sizes of explanations, correct explanations, and incorrect explanations.
    """
    # Create a bar plot
    x = range(len(sizes))
    # plt.bar(x, sizes, label='Total Explanations', alpha=0.6)
    plt.bar(x, correct_sizes, label="Correct Explanations", alpha=0.6)
    plt.bar(x, incorrect_sizes, label="Incorrect Explanations", alpha=0.6)

    # Add labels and title
    plt.xlabel("Class Index")
    plt.ylabel("Number of Explanations")
    plt.title(f"Explanations for {paradigm.capitalize()} Paradigm")
    plt.xticks(x, [str(i) for i in x])
    plt.legend()
    plt.show()


def calculate_stats(sizes, correct_sizes, incorrect_sizes):
    """Calculates mean statistics."""
    _stats = {}
    _stats["Mean Size"] = np.mean(sizes) if sizes else 0
    # _stats["Standard Deviation"] = np.std(sizes) if sizes else 0
    # _stats["Minimum Size"] = np.min(sizes) if sizes else 0
    # _stats["Maximum Size"] = np.max(sizes) if sizes else 0
    _stats["Correct Samples"] = np.mean(correct_sizes) if correct_sizes else 0
    _stats["Incorrect Samples"] = np.mean(incorrect_sizes) if incorrect_sizes else 0
    return _stats


def generate_latex_table(results, caption="Summary of Mean Sizes", label="tab:mean_sizes_summary"):
    """
    Generates LaTeX code for a table summarizing the explanation size results.

    Args:
        results (dict): A dictionary where keys are paradigm names (str)
                        and values are dictionaries containing stats
                        {'Mean Size': float, 'Correct Samples': float, 'Incorrect Samples': float}.
                        An OrderedDict is recommended to control column order.
        caption (str): The caption for the LaTeX table.
        label (str): The label for the LaTeX table.

    Returns:
        str: LaTeX code for the table.
    """
    if not results:
        return "% No results to generate table."

    # Define the order of rows and their display names
    row_keys_map = collections.OrderedDict(
        [("Mean Size", "Mean Size"), ("Correct Samples", "Correct Samples"), ("Incorrect Samples", "Incorrect Samples")]
    )

    paradigms = list(results.keys())

    # --- Build LaTeX String ---
    latex_string = "\\begin{table}[ht!]\n"
    latex_string += "    \\centering\n"
    latex_string += f"    \\label{{{label}}}\n"

    # Define column format - l for first column, c for the rest
    # Handle the vertical bar specifically if 'Top_k (Adjusted)' exists
    col_format = "l"
    num_data_cols = len(paradigms)
    if "Top_k (Adjusted)" in paradigms:
        # Assumes 'Top_k (Adjusted)' is the last column if present
        col_format += "c" * (num_data_cols - 1) + "|c"
    else:
        col_format += "c" * num_data_cols

    latex_string += f"    \\begin{{tabular}}{{{col_format}}}\n"
    latex_string += "        \\toprule\n"

    # Header row
    header_row = "        "  # Initial space for the first column alignment
    for _paradigm in paradigms:
        # Escape underscores for LaTeX
        safe_paradigm_name = _paradigm.replace("_", "\\_").capitalize()
        header_row += f"& {safe_paradigm_name} "
    header_row += "\\\\\n"
    latex_string += header_row
    latex_string += "        \\midrule\n"

    # Data rows
    first_data_row = True
    for key, display_name in row_keys_map.items():
        if key == "Correct Samples" and not first_data_row:  # Add midrule before Correct/Incorrect if needed
            latex_string += "        \\midrule\n"  # As per example, only one midrule after header

        row = f"        {display_name} "
        for _paradigm in paradigms:
            value = results[_paradigm].get(key, 0)  # Get value, default to 0 if missing
            row += f"& {value:.0f} "  # Format as integer
        row += "\\\\\n"
        latex_string += row
        first_data_row = False

    latex_string += "        \\bottomrule\n"
    latex_string += "    \\end{tabular}\n"
    latex_string += f"    \\caption{{{caption}}}\n"
    latex_string += "\\end{table}"

    return latex_string


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse explanation files and generate statistics.")
    parser.add_argument("--save", action="store_true", help="Save the plots.")
    parser.add_argument("--latex", action="store_true", help="Generate LaTeX table.")
    parser.add_argument("--max-index", type=int, default=-1, help="Maximum index to parse (default: -1).")
    parser.add_argument("--data-folder", type=str, default="", help="Folder containing the data files.")
    parser.add_argument("-d", "--dataset", type=str, default="cub200", help="Dataset name for parsing indices.")
    parser.add_argument("-m", "--model-arch", type=str, default="protopnet", help="Model architecture name for parsing indices.")
    parser.add_argument("-s", "--seed", type=int, default=42, help="Random seed for reproducibility.")
    # parser.add_argument("--not-all-indices", action="store_true", help="If set, use the indices from the file instead of all indices.")
    parser.add_argument("--use-indices", action="store_true", help="If set, use the indices from the file instead of all possible indices.")
    args = parser.parse_args()
    # if the data folder is not provided, create it using the dataset and model_arch and seed
    CWD = Path.cwd()
    if not args.data_folder:
        args.data_folder = f"{CWD}/logs/{args.model_arch}_{args.dataset}_{args.seed}/final/explanations/"
    
    # Define the paradigms and their respective indices
    paradigms_list = [
        "triangle",
        "hypersphere",
        "top_k",
    ]
    # idx_file = "./random_indices.txt"
    # idx_file = os.path.join(args.data_folder, "random_indices.txt")
    idx_file = Path(args.data_folder).parent / "random_indices.txt"
    indices = []
    # Read the indices from the file - Dummy implementation if file doesn't exist
    datapath = Path(args.data_folder)
    print(f"Data Folder: {datapath}") # .../model_folder/final/explanations/ 
    model_folder = datapath.parent.parent.name # e.g., protopnet_cub200_42
    model_full_path = datapath.parent  # e.g., /home/jsoria/cabrnet/cabrnet/logs/protopnet_cub200_42/final
    print(f"Model Folder: {model_folder}")  # e.g., protopnet_cub200_42
    # Load model and check how many prototypes and how many classes it has
    from cabrnet.archs.generic.model import CaBRNet
    config_file = str(model_full_path / "model_arch.yml")
    print(f"Config File: {config_file}")
    if not Path(config_file).exists():
        raise FileNotFoundError(f"Model architecture file {config_file} does not exist.")
    state_dict_path = str(model_full_path / "model_state.pth")
    print(f"State Dict Path: {state_dict_path}")
    if not Path(state_dict_path).exists():
        raise FileNotFoundError(f"Model state file {state_dict_path} does not exist.")
    model = CaBRNet.build_from_config(
        config=config_file,
        state_dict_path=state_dict_path,
        seed=args.seed,
    )
    from cabrnet.core.utils.data import DatasetManager as DM
    # Load the model to get the latent space size
    model.eval()  # Set the model to evaluation mode
    data_file = str(model_full_path / "dataset.yml")
    test_data = DM.get_dataloaders(config=data_file)["test_set"]
    one_sample_test_data = torch.utils.data.DataLoader(
        test_data.dataset, batch_size=1, shuffle=False, num_workers=1
    )
    # Get the latent space size
    img, _ = next(iter(one_sample_test_data))
    with torch.no_grad():
        batch_latent_space = model.extractor(img).shape  # (B, D, H, W)
        h, w = batch_latent_space[2], batch_latent_space[3]
        factor = h * w  # Assuming the latent space is flattened to a single dimension
        print(f"Latent Space Size: {batch_latent_space}, Factor: {factor}")
    num_prototypes = model.classifier.num_prototypes
    num_classes = model.classifier.num_classes
    
    print(f"Number of Prototypes: {num_prototypes}")
    print(f"Number of Classes: {num_classes}")
    print(f"Average Prototypes per Class: {num_prototypes / num_classes:.2f}")
    
    print(f"Model Architecture: {model.__class__.__name__}")
    print(f"Model Folder: {model_folder}")
    if not args.dataset:
        # If dataset is not provided, extract it from the model folder name
        # Assuming the format is 'model_arch_extra_dataset_seed'
        args.dataset = model_folder.split("_")[-2]
    else:
        dataset = args.dataset
    print(f"Dataset: {dataset}")
    try:
        with open(idx_file, "r") as file:
            lines = file.readlines()
            for line in lines:
                indices.append(int(line.strip()))
    except FileNotFoundError:
        print(f"Warning: {idx_file} not found. Using dummy indices.")
        
        if dataset == "cars":
            indices = list(range(8041))
        elif dataset == "cub200":
            indices = list(range(5794))
        # elif dataset == "mnist":
        elif dataset == "flowers102":
            indices = list(range(6149))
        # elif dataset == "cifar100":
        # elif dataset == "cifar10":
        # elif dataset == "oxford_iiit_pet":
        else:
            # Default to a dummy range if no specific dataset is matched
            print("Using default dummy indices (0-9).")
            indices = list(range(10))
        # indices = list(range(10))  # Dummy indices

    # Use OrderedDict to maintain the order of paradigms for table columns
    all_results = collections.OrderedDict()
    
    # Parse and calculate stats for each paradigm
    for paradigm in paradigms_list:
        sizes, correct_sizes, incorrect_sizes = parse_paradigm(paradigm, indices, data_folder=args.data_folder, not_all_indices=args.use_indices)
        print(f"Parsed {len(sizes)} sizes, {len(correct_sizes)} correct sizes, and {len(incorrect_sizes)} incorrect sizes for paradigm '{paradigm}'.")
        # Basic sanity check (optional, but good practice)
        if sizes:
            reconstructed_sizes = [c + i for c, i in zip(correct_sizes, incorrect_sizes)]
            # Note: This assertion might fail if means are used directly without original lists
            # assert len(sizes) == len(reconstructed_sizes), f"Sizes do not match. {len(sizes)} != {len(reconstructed_sizes)}"
            # assert np.sum(sizes) == np.sum(reconstructed_sizes), f"Sizes do not match: {np.sum(sizes)} != {np.sum(reconstructed_sizes)}"
        else:
            print(f"Warning: No data parsed for paradigm {paradigm}")
            continue  # Skip if parsing failed
        
        # Truncate indices if max-index is set
        if args.max_index > 0:
            indices = [idx for idx in indices if idx < args.max_index]
            sizes = [size for idx, size in zip(indices, sizes) if idx < args.max_index]
            correct_sizes = [size for idx, size in zip(indices, correct_sizes) if idx < args.max_index]
            incorrect_sizes = [size for idx, size in zip(indices, incorrect_sizes) if idx < args.max_index]
            
        # Calculate stats
        stats = calculate_stats(sizes, correct_sizes, incorrect_sizes)
        all_results[paradigm] = stats

        # Handle 'top_k' adjustment specifically
        if paradigm == "top_k":
            adjusted_stats = {}
            # find a way to find the correct factor based on the latent space size
            # factor = 49  # Example factor, adjust as needed
            for key, value in stats.items():
                adjusted_stats[key] = value * factor
            # # Add adjusted stats to results
            # adjusted_stats["Mean Size"] = stats.get("Mean Size", 0) * factor
            # adjusted_stats["Correct Samples"] = stats.get("Correct Samples", 0) * factor
            # adjusted_stats["Incorrect Samples"] = stats.get("Incorrect Samples", 0) * factor
            all_results["Top_k (Adjusted)"] = adjusted_stats  # Add as a new entry

    # --- Generate and print the LaTeX table ---
    if args.latex:
        latex_output = generate_latex_table(all_results)
        print("\nGenerated LaTeX Code:\n")
        print(latex_output)

    # --- Optional: Print individual stats as before for verification ---
    print("\n" + "%" * 50)
    print("Individual Statistics (for verification):")
    for paradigm, stats in all_results.items():
        print(f"Paradigm: {paradigm}")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}")
        # print(f"  Mean Size: {stats.get('Mean Size', 0):.2f}")
        # print(f"  Mean Correct Size: {stats.get('Correct Samples', 0):.2f}")
        # print(f"  Mean Incorrect Size: {stats.get('Incorrect Samples', 0):.2f}")
    print("%" * 50)
