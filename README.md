# Neural Machine Unranking

This repository provides the official implementation of the paper
**"Neural Machine Unranking"**, published in\
**IEEE Transactions on Neural Networks and Learning Systems (IEEE
TNNLS)**.

The work systematically studies **machine unlearning in neural
information retrieval (IR)**, including: - A formal definition of
*machine unranking* for neural IR systems, - Unlearning principles and
evaluation criteria, - A comprehensive comparison of representative
unlearning baselines, - A novel unlearning objective, **Contrastive and
Consistent Loss (CoCoL)**.

------------------------------------------------------------------------

## Overview

Machine unlearning aims to selectively remove the influence of specific
training instances from a trained model **without full retraining**.\
In neural information retrieval, this problem is particularly
challenging due to the tight coupling between queries, documents, and
ranking objectives.

This repository: - Reproduces **six representative machine unranking
baselines**, - Supports **four widely adopted neural ranking models**, -
Evaluates unranking behavior on **two standard IR benchmarks**, -
Implements the proposed **Contrastive and Consistent Loss (CoCoL)** in
two refined variants.

------------------------------------------------------------------------

## Supported Datasets and Models

### Datasets

-   **MS MARCO**
-   **TREC CAR Y3**

### Neural Ranking Models

-   **BERTcat**
-   **BERTdot**
-   **ColBERT**
-   **PARADE**

------------------------------------------------------------------------

## Repository Structure

    .
    ├── Conf/                 # Configuration files for datasets and ranking models
    ├── Data/                 # Dataset storage
    ├── models/               # Neural ranking model implementations
    ├── models_saved/         # Saved model checkpoints
    ├── results/              # Prediction outputs and experimental results
    │
    ├── ranking_dataset.py    # Dataset construction and preprocessing
    ├── task_utils.py         # Utility functions for unranking tasks
    ├── unranking_methods.py  # Implementations of machine unranking methods
    ├── train.py              # Training, backpropagation, gradient manipulation, and testing
    ├── unranking_task.py     # Workflows for different machine unranking strategies
    ├── task_launcher.py      # Task configuration and execution entry point
    ├── task_eval.py          # Evaluation scripts
    ├── ssd.py                # Selective Synaptic Dampening (SSD) implementation

------------------------------------------------------------------------

## Implemented Machine Unranking Methods

This repository reproduces the following **six baseline machine
unranking methods** for neural information retrieval:

-   `amnesiac`
-   `retrain`
-   `catastrophic`
-   `NegGrad`
-   `BadTeacher`
-   `SSD` (Selective Synaptic Dampening)

> **Note**\
> The SSD implementation is adapted from the official repository:\
> https://github.com/if-loops/selective-synaptic-dampening/tree/main/src

------------------------------------------------------------------------

## Contrastive and Consistent Loss (CoCoL)

We propose **Contrastive and Consistent Loss (CoCoL)**, a novel
unlearning objective specifically designed for **neural machine
unranking**.

### CoCoL Variants

1.  **CoCoL (v1)**\
    Introduced in the author's PhD thesis (Chapter 5):\
    https://repository.lboro.ac.uk/articles/thesis/Advancing_neural_machine_continual_learning_and_unlearning_for_language_models_in_information_retrieval_systems/27907359

2.  **CoCoL (v2)**\
    Presented in the IEEE TNNLS journal article.\
    This version further **reduces dependence on empirically tuned
    hyperparameters**, improving robustness and practical applicability.

------------------------------------------------------------------------

## Data and Pretrained Models

We provide **example machine unranking datasets** derived from **MS MARCO** and **TREC CAR**, together with **pretrained neural ranking models** trained on the corresponding *original datasets*.  
These pretrained models can be **directly used as starting points for machine unranking experiments**, without requiring additional training from scratch.

All example datasets and pretrained model checkpoints are available via **OneDrive**:

🔗 https://1drv.ms/f/c/00c07038f4fdc681/IgC_c3qKUQxvS7Z3rbjxwDf0AWpElzh7agp6QIvqvwBVLYE

In addition, we provide a **separate repository** dedicated to constructing **custom machine unranking datasets**, including detailed instructions and scripts for preparing your own data:

🔗 https://github.com/JingruiHou/unranking_datasets


## Usage

Experiments are configured and launched via:

``` bash
python task_launcher.py
```

Please refer to the configuration files in the `Conf/` directory to
specify: - Dataset selection, - Neural ranking model, - Machine
unranking method, - Training and evaluation hyperparameters.

------------------------------------------------------------------------

## Citation

If you use this codebase in your research, please cite the following
works:

``` bibtex
@phdthesis{hou2024advancing,
  title={Advancing neural machine continual learning and unlearning for language models in information retrieval systems},
  author={Hou, Jingrui},
  year={2024},
  school={Loughborough University}
}

@ARTICLE{Hou2026Neural,
  author={Hou, Jingrui and Finke, Axel and Cosma, Georgina},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Neural Machine Unranking},
  year={2025},
  pages={1--12},
  doi={10.1109/TNNLS.2025.3639808}
}
```

------------------------------------------------------------------------

## Contact

For questions, discussions, or collaboration inquiries, please contact:

**Jingrui Hou**\
📧 jhou.research@outlook.com

------------------------------------------------------------------------

## License

This project is released **for research purposes only**.\
Please consult the licenses of the corresponding datasets for usage
restrictions.
