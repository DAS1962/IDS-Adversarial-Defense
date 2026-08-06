
import torch
import numpy as np
import pandas as pd
import sklearn
import matplotlib
import yaml
from pathlib import Path
import sys


SEPARATOR = "-" * 60


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{title}")
    print(SEPARATOR)


def check_versions() -> None:
    """Print versions of core libraries."""
    print_header("Library versions")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  PyTorch      : {torch.__version__}")
    print(f"  NumPy        : {np.__version__}")
    print(f"  Pandas       : {pd.__version__}")
    print(f"  Scikit-learn : {sklearn.__version__}")
    print(f"  Matplotlib   : {matplotlib.__version__}")


def check_device() -> None:
    """Print information about compute device."""
    print_header("Compute device")
    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"  CUDA available : {cuda_available}")
    print(f"  Device in use  : {device}")


def check_pytorch() -> None:
    """Run a basic PyTorch tensor operation."""
    print_header("PyTorch functional test")
    x = torch.randn(3, 4)
    y = torch.randn(4, 5)
    z = x @ y
    print(f"  Matrix multiplication OK - result shape: {tuple(z.shape)}")


def check_config(config_path: Path) -> None:
    """Load and validate the YAML configuration file."""
    print_header("Configuration file")
    if not config_path.exists():
        print(f"  ERROR: {config_path} not found")
        return

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        print(f"  WARNING: {config_path} is empty")
        return

    print(f"  File     : {config_path}")
    print(f"  Sections : {len(config)}")
    for section in config.keys():
        print(f"    - {section}")


def check_project_structure() -> None:
    """Verify that all expected top-level directories exist."""
    print_header("Project structure")
    required_dirs = [
        "src", "src/data", "src/models", "src/attacks",
        "src/defenses", "src/utils",
        "data", "data/raw", "data/processed",
        "results", "results/logs", "results/checkpoints", "results/figures",
        "configs", "notebooks", "scripts", "tests",
    ]
    missing = []
    for d in required_dirs:
        if Path(d).is_dir():
            print(f"  OK     {d}/")
        else:
            print(f"  MISSING {d}/")
            missing.append(d)

    if missing:
        print(f"\n  {len(missing)} directory(ies) missing.")


def main() -> None:
    """Run the full environment check."""
    print(SEPARATOR)
    print("Environment sanity check")
    print(SEPARATOR)

    check_versions()
    check_device()
    check_pytorch()
    check_config(Path("configs/config.yaml"))
    check_project_structure()

    print(f"\n{SEPARATOR}")
    print("Check complete.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()