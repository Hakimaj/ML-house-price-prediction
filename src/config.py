"""Central place for paths and constants, so nothing is hard-coded elsewhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "house_price_model.joblib"
METRICS_PATH = ROOT / "models" / "metrics.json"
RANDOM_STATE = 42 # makes every random step repeatable
TARGET = "MedHouseVal" # the column we want to predict (units: $100,000)
