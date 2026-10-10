"""Feature engineering and the full preprocessing + model pipeline."""
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

# The eight raw input columns, in the order the model was trained on.
RAW_FEATURES = [
 "MedInc", "HouseAge", "AveRooms", "AveBedrms",
 "Population", "AveOccup", "Latitude", "Longitude",
]


def add_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create new columns from existing ones. Works on a copy, never edits the input."""
    X = X.copy()
    X["BedroomRatio"] = X["AveBedrms"] / X["AveRooms"] # share of rooms that are bedrooms
    X["RoomsPerPerson"] = X["AveRooms"] / X["AveOccup"] # space per resident
    X["LogPopulation"] = np.log1p(X["Population"]) # tames the long right tail
    return X


def build_pipeline(model) -> Pipeline:
    """Chain: add features -> scale -> model. One object, so train and predict stay identical."""
    return Pipeline([
    ("features", FunctionTransformer(add_features)), # FunctionTransormer wraps an ordinary function so it can be a Pipeline step.
    ("scale", StandardScaler()), # rescales eert column to mean 0 and std 1.
    ("model", model), # The algorithm passed to build_pipeline like (linear regression...)
    ])