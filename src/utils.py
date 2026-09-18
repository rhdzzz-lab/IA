"""
Funciones auxiliares del proyecto.
"""

import random
import numpy as np
import torch

from src.config import (
    DATA_DIR,
    CSV_DIR,
    JSONL_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    CONFIG,
)


def set_seed(seed=None):
    """
    Fija la semilla para obtener resultados reproducibles.
    """
    if seed is None:
        seed = CONFIG["random_seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """
    Detecta automáticamente el mejor dispositivo disponible.
    """

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_directories():
    """
    Crea automáticamente las carpetas necesarias.
    """

    directories = [
        DATA_DIR,
        CSV_DIR,
        JSONL_DIR,
        MODELS_DIR,
        REPORTS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
