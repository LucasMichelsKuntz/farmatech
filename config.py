from pathlib import Path

ROOT       = Path(__file__).parent
CSV_PATH   = ROOT / "data" / "crop_dataset.csv"
DB_PATH    = ROOT / "farmtech.db"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "models.joblib"
