
# IDS Adversarial Defense Framework

Reproduction and extension of:

**Awad, Z., Zakaria, M., & Hassan, R. (2025).** "An Enhanced Ensemble Defense Framework for Boosting Adversarial Robustness of Intrusion Detection Systems." *Scientific Reports*, 15, 14177.
DOI: [10.1038/s41598-025-94023-z](https://doi.org/10.1038/s41598-025-94023-z)

## Context

Research internship project.
Start date: July 2026.

## Objective

Reproduce and evaluate an ensemble-based defense framework against adversarial attacks on Deep Learning-based Intrusion Detection Systems (IDS).

## Project structure

```
IDS-Adversarial-Defense/
├── src/                 Source code
│   ├── data/            Data loading and preprocessing
│   ├── models/          DNN architectures
│   ├── attacks/         Adversarial attack implementations
│   ├── defenses/        Defense mechanism implementations
│   └── utils/           Utility functions
├── notebooks/           Jupyter notebooks for exploration
├── scripts/             Executable scripts
├── configs/             YAML configuration files
├── data/                Datasets (not versioned)
├── results/             Outputs (logs, checkpoints, figures)
└── tests/               Unit tests
```

## Installation

Requires Python 3.12 on Linux.

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate

# Install PyTorch (CPU-only for local development)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
pip install -r requirements.txt

# Install torchattacks (--no-deps to avoid the requests version conflict)
pip install torchattacks==3.5.1 --no-deps
```

## Adversarial attacks implemented

- FGSM (Goodfellow et al., 2014)
- BIM (Kurakin et al., 2016)
- PGD (Madry et al., 2017)
- DeepFool (Moosavi-Dezfooli et al., 2015)
- JSMA (Papernot et al., 2015)
- C&W (Carlini & Wagner, 2016)

## Defense mechanisms

- Adversarial Training (AT)
- Gaussian Augmentation (GA)
- Label Smoothing (LS)
- Denoising Autoencoder (DAE)

Ensemble aggregation: Majority Voting and Weighted Average, both optimized via Bayesian Optimization.

## Datasets

- CIC-IDS 2017: https://www.unb.ca/cic/datasets/ids-2017.html
- CIC-IDS 2018: https://www.unb.ca/cic/datasets/ids-2018.html

## Reference results (from the paper, CIC-IDS 2017)

| Configuration | Accuracy |
|---|---:|
| Baseline (clean data) | 98.11% |
| Baseline under C&W attack | 36.00% |
| Label Smoothing (standalone) | 85.90% |
| Optimized ensemble (Majority Voting) | 87.49% |

## License

Academic project. Research use only.