"""Train, tune, evaluate and save the final model. Run: python -m src.train"""
import json

import joblib
import numpy as np
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV

from .config import METRICS_PATH, MODEL_PATH, RANDOM_STATE
from .data import load_data, split_data
from .features import build_pipeline

def main(n_iter: int = 30) -> None:
 X, y = load_data()
 X_train, X_test, y_train, y_test = split_data(X, y)
 
 pipeline = build_pipeline(HistGradientBoostingRegressor(max_iter=500, random_state=RANDOM_STATE))
 param_distributions = {
 "model__learning_rate": loguniform(0.02, 0.3),
 "model__max_leaf_nodes": randint(15, 80),
 "model__min_samples_leaf": randint(10, 80),
 "model__l2_regularization": loguniform(1e-3, 10),
 }
 search = RandomizedSearchCV(
 pipeline,
 param_distributions=param_distributions,
 n_iter=n_iter,
 cv=KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
 scoring="neg_root_mean_squared_error",
 random_state=RANDOM_STATE,
 n_jobs=-1,
 verbose=1,
 )
 search.fit(X_train, y_train)
 best = search.best_estimator_
 
 # The one and only look at the test set:
 pred = best.predict(X_test)
 metrics = {
 "cv_rmse": float(-search.best_score_),
 "test_rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
 "test_mae": float(mean_absolute_error(y_test, pred)),
 "test_r2": float(r2_score(y_test, pred)),
 "best_params": {k: (float(v) if isinstance(v, float) else int(v))
 for k, v in search.best_params_.items()},
 }
 
 MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
 joblib.dump(best, MODEL_PATH)
 METRICS_PATH.write_text(json.dumps(metrics, indent=2))
 print(json.dumps(metrics, indent=2))
 print(f"Saved model to {MODEL_PATH}")
 
 
if __name__ == "__main__": # required on Windows when n_jobs=-1 starts worker processes
 main()
