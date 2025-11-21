<div align="center">

# Formal Abductive Explanations for Prototype-Based Networks

[![Conference](https://img.shields.io/badge/AAAI-26-blue)](https://aaai.org/conference/aaai/aaai-26/)
[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2511.16588)
[![Zenodo](https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.16707325-blue)](https://zenodo.org/records/16707325)
[![License](https://img.shields.io/badge/License-LGPL_v2.1-blue.svg)](LICENSE)

<br>

🎉 **Accepted at AAAI-26 (Main Technical Track)** 🎉

</br>

### Authors

[**Jules Soria**](https://scholar.google.com/citations?user=1Qctec0AAAAJ) ✉️, 
[Zakaria Chihani](https://scholar.google.com/citations?user=mgzCh30AAAAJ), 
[Julien Girard-Satabin](https://scholar.google.com/citations?user=erWN5TwAAAAJ), 
[Alban Grastien](https://scholar.google.com/citations?user=87u0x6UAAAAJ), 
[Romain Xu-Darme](https://scholar.google.com/citations?user=QB4YkI0AAAAJ), 
[Daniela Cancila](https://scholar.google.com/citations?user=ucNJ23sAAAAJ)

<small>✉️ Corresponding Author: firstname.lastname [at] cea [dot] fr</small>

</div>

---

## Environment settings

This project is tested under the following environment setting:
- OS: Ubuntu 24.04.2
- GPU: RTX 2000 Ada Generation Laptop GPU or Tesla V100
- Cuda: 12.0, Cudnn: v12.4
- Python: 3.10.13
- PyTorch: 2.1.2
- Torchvision: 0.16.2
- CaBRNet: 1.1

From a clean environment, running this should be enough to guarantee a reproductible setting:

<!-- ```bash
pip install cabrnet  # includes all other requirements
pip install psutil  # for memory tracking
```  -->

```bash
pip install -r requirements.txt
```



## Installation and Downloads
From `torchvision` we used multiple datasets, and for each one, trained a ProtoPNet model using `cabrnet train` 

You can download all trained models in an anonymous Zenodo repository here : [ProtoPNet Models](https://zenodo.org/records/16707325?token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc1NDA5MjQwNSwiZXhwIjoxNzY5NTU4Mzk5fQ.eyJpZCI6ImZiNTcwOTk5LWY4ZTUtNDVjZS1iYWQ0LWJhMmMxMjg4Njc2YiIsImRhdGEiOnt9LCJyYW5kb20iOiI5NDNiMDlhMGVlYjJhYjAxODViMDJjMTU2MjlmZmExYiJ9.4XskPhagYJFHV4-DkAdZwGTyN0RFrmZH2YVda5cV4DuGsCbrM5XRjPiyRDUs2bIOXKtTr20kqqazL1A_XG4v7w)

All the model folders will follow the same naming style: `<arch>_<dataset>_<seed>/final/`
Please put them in a subfolder `/logs` such that the overall structure of the project folder is :

```bash
.
├── data                           # folder with all datasets, each added by torchvision and cabrnet
│   ├── cars                       #
│   ├── cifar-10-batches-py        #
    ...                            #
├── graph_results.py
├── hypersphere_calc_test.py
├── logs                           # folder where you will dump all models downloaded from zenodo
│   └── protopnet_cifar10_42       #
│       └── final
    ...
│   └── protopnet_stanford_cars_42 #
│       └── final
├── README.md
├── run_formal_exp.py
├── subset_minimal_axp_base.py
├── subset_minimal_axp_spatial.py
├── subset_minimal_axp_topk.py
├── utils_plus.py
└── utils.py
```

Each individual 'final' model folder will have the following structure:

```bash
.
├── dataset.yml
├── model_arch.yml
├── model_state.pth
├── projection_info.csv
├── prototypes.pth
├── random_indices.txt
├── state.pickle
└── training.yml
```

In each configuration file, there are references to `torchvision` datasets and models, they will be downloaded as needed during the generating phase of ***Abductive Latent Explanations***, you do not need to manually download them.
> Two notable exceptions are the fine-grained `CUB_200_2011` and `Stanford Cars` datasets, which are necessary to reproduce the relevant experiments of the paper, and should be downloaded using the `cabrnet` command `download_datasets`.


## Running Experiments

In order to generate ALEs, you need to use the following command:

```bash
python run_formal_exp.py --paradigm <paradigm> --dataset <data> --seed <seed> --model-arch protopnet
```

### Example:

```bash
python run_formal_exp.py --paradigm triangle --dataset cifar10 --seed 42 --model-arch protopnet
```

This command performs the ALE generation process with the prescribed model (only ProtoPNet so far), dataset, and seed. If you type `python run_formal_exp.py --help` you will notice a `all_indices` argument. We recommend activating it only for the `top_k` paradigm since spatial constraints tend to take some time to generate.

Then, to visualise the results for each model, use the following command:

```bash
python graph_results.py --dataset <dataset> --seed <seed> --model-arch protopnet
```

### Example:
```bash
python graph_results.py --dataset cifar10 --seed 42 --model-arch protopnet
```

After computing the explanations for all paradigms, the output of the example command should be:

```bash
...
Latent Space Size: torch.Size([1, 128, 1, 1]), Factor: 1
Number of Prototypes: 100
Number of Classes: 10
Average Prototypes per Class: 10.00
Model Architecture: ProtoPNet
Model Folder: protopnet_cifar10_42
Dataset: cifar10
Parsed 10000 sizes, 8317 correct sizes, and 1683 incorrect sizes for paradigm 'triangle'.
Parsed 10000 sizes, 8317 correct sizes, and 1683 incorrect sizes for paradigm 'hypersphere'.
Parsed 10000 sizes, 8317 correct sizes, and 1683 incorrect sizes for paradigm 'top_k'.

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Individual Statistics (for verification):
Paradigm: triangle
  Mean Size: 22.30
  Correct Samples: 6.58
  Incorrect Samples: 100.00
Paradigm: hypersphere
  Mean Size: 24.25
  Correct Samples: 8.92
  Incorrect Samples: 100.00
Paradigm: top_k
  Mean Size: 41.10
  Correct Samples: 36.89
  Incorrect Samples: 61.95
Paradigm: Top_k (Adjusted)
  Mean Size: 41.10
  Correct Samples: 36.89
  Incorrect Samples: 61.95
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
```

> Note:
> In the paper, we used `all_indices` for the paradigm top-$k$ and for all paradigms for CIFAR datasets.

---

> Note:
> Practitioners who try to reproduce the exact results might encounter small differences due to the hardware architecture used when computing ALEs, but we made efforts to remove them as much as we could.

---

## Recommendations

Because some experiments take a lot of time, we recommend reproducing as much as you can on small resolution datasets like CIFAR10 and MNIST, as experiments should be very quick for all paradigms. Furthermore, computing ALEs with the top-$k$ paradigm for other larger datasets should not be too much time-consuming, and we recommend approving our result by running them on as many datasets as possible.

***

## 📄 How to Cite

If you use this code in your research, please cite our paper:


**BibTeX:**
```bibtex
@inproceedings{soria2026formal,
  title     = {Formal Abductive Explanations for Prototype-Based Networks},
  author    = {Soria, Jules and Chihani, Zakaria and Girard-Satabin, Julien and Grastien, Alban and Xu-Darme, Romain and Cancila, Daniela},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  year      = {2026}
}
```

***
## ⚖️ License

This project is licensed under the **GNU Lesser General Public License v2.1** - see the [LICENSE](LICENSE) file for details.

