"""
Constantes globales del proyecto.
"""

# Modelo base de Hugging Face
MODEL_NAME = "bert-base-multilingual-cased"

# Semilla para reproducibilidad
RANDOM_SEED = 42

# Longitud máxima del texto
MAX_LENGTH = 128

# Parámetros de entrenamiento
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5

# División del dataset
TEST_SIZE = 0.20

# Directorios
DATA_DIR = "data"
CSV_DIR = "data/csv"
JSONL_DIR = "data/jsonl"

MODELS_DIR = "models"
REPORTS_DIR = "reports"
