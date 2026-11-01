import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = BASE_DIR / "results"

# Directories to ensure exist
directories = [
    DATA_DIR,
    RAW_DATA_DIR, 
    PROCESSED_DATA_DIR, 
    RESULTS_DIR,
    RESULTS_DIR / "tables", 
    RESULTS_DIR / "figures", 
    RESULTS_DIR / "models"
]

# Safe creation of directories
for p in directories:
    if not p.exists():
        p.mkdir(parents=True, exist_ok=True)

# Global Reproducibility
SEED = 42

# Default Data Cleaning Configs
K_CORE = 5

# Ranking Evaluation Settings
TOP_K_LIST = [5, 10, 20]
NUM_NEGATIVES = 99