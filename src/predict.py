"""Load the saved model once and turn a dict of house features into a price."""
import joblib
import pandas as pd

from .config import MODEL_PATH
from .features import RAW_FEATURES

_model = None


def get_model():
 global _model
 if _model is None:
    _model = joblib.load(MODEL_PATH)
 return _model


def predict_value(features: dict) -> float:
 """Return the predicted median house value in units of $100,000."""
 row = pd.DataFrame([features], columns=RAW_FEATURES) # same names and order as training
 return float(get_model().predict(row)[0])