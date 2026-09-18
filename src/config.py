"""
Configuración general del proyecto.
"""

from pathlib import Path
from src.constants import (
    MODEL_NAME,
    RANDOM_SEED,
    MAX_LENGTH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
)

# ===========================
# Directorio raíz del proyecto
# ===========================
BASE_DIR = Path(__file__).resolve().parent.parent

# ===========================
# Carpetas principales
# ===========================
DATA_DIR = BASE_DIR / "data"
CSV_DIR = DATA_DIR / "csv"
JSONL_DIR = DATA_DIR / "jsonl"

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

# ===========================
# Configuración del modelo
# ===========================
CONFIG = {
    "model_name": MODEL_NAME,
    "random_seed": RANDOM_SEED,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
}
