import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.features import RAW_FEATURES, add_features, build_pipeline


def make_frame(n=200):
 rng = np.random.default_rng(0)
 return pd.DataFrame({c: rng.uniform(1, 10, n) for c in RAW_FEATURES})


def test_add_features_adds_columns_and_keeps_input_unchanged():
 X = make_frame()
 out = add_features(X)
 assert {"BedroomRatio", "RoomsPerPerson", "LogPopulation"} <= set(out.columns)
 assert list(X.columns) == RAW_FEATURES # the original was not modified
 assert not out.isna().any().any()
 
 
def test_pipeline_fits_and_predicts_right_shape():
 X = make_frame()
 y = np.random.default_rng(1).uniform(0.5, 5, len(X))
 model = build_pipeline(LinearRegression()).fit(X, y)
 assert model.predict(X.head(3)).shape == (3,)
