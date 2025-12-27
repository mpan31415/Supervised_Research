# Supervised Research (Law, Economics and Data Science)

This repository contains code, scripts, and resources for performing representation learning on a large dataset of gravestone imagery. All files are included to support reproducible experiments, dataset management, model training, evaluation, and result visualization.


## Table of Contents

- [Supervised Research (Law, Economics and Data Science)](#supervised-research-law-economics-and-data-science)
  - [Table of Contents](#table-of-contents)
  - [Repository Structure](#repository-structure)
  - [Installation](#installation)
    - [Prerequisites](#prerequisites)
    - [Setup Instructions](#setup-instructions)
  - [Usage](#usage)
    - [Train a Model](#train-a-model)
    - [Evaluate a Model](#evaluate-a-model)
  - [Visualization](#visualization)


## Repository Structure

Here, we summarize the contents of each of the directories, and the purpose of selected files which may particularly useful.

```
Supervised_Research/
├── checkpoints/              # Checkpoints of trained models
├── database/                 # Database utility and test scripts
├── dataset/                  # Dataset utility and test scripts
├── plotting/                 # Scripts for plotting and visualizations
├── scripts/                  # Training, evaluation, and preprocessing scripts
    └── cls_attn_viz.py         # for creating the CLS vector attention maps
    └── eval_latent.py          # runs PCA and t-SNE on the small dataset
    └── models.py               # modified from the vit-pytorch implementation
    └── train_distributed.py    # main script for distributed training across GPUs
├── slurm_scripts/            # SLURM job scripts for HPC environments
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```


## Installation

### Prerequisites

- Python 3.7 or higher
- (Optional) CUDA-compatible GPU for accelerated training

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/mpan31415/Supervised_Research.git
   cd Supervised_Research
   ```

2. Create and activate a virtual environment:
    ```bash
    conda env create -f environment.yml
    conda activate env_gravestones
    ```


## Usage

### Train a Model
```bash
# python (execute from /scripts directory)
torchrun --nproc_per_node=8 train_distributed.py train.yaml

# slurm (execute from /slurm_scripts directory)
sbatch train_model_distributed.slurm
```

### Evaluate a Model
```bash
# python (execute from /scripts directory)
python eval_latent.py
python cls_attn_viz.py
python run_probing.py
python run_probing_ranking.py
python run_partial_finetune.py
python run_partial_finetune_ranking.py

# slurm (execute from /slurm_scripts directory)
sbatch eval_latent.slurm
sbatch run_encoder_eval.slurm
```


## Visualization

Generate visualization and figures with:
```bash
# python (execute from /ploting directory)
python classification_acc.py
python confusion_matrix.py
python year_bin_acc.py
```
