"""Loading the data and splitting it into training and test sets."""
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

from .config import RANDOM_STATE, TARGET

def load_data():
 """Return features X (a DataFrame) and target y (a Series)."""
 df = fetch_california_housing(as_frame=True).frame
 X = df.drop(columns=TARGET)
 y = df[TARGET]
 return X, y

def split_data(X, y, test_size=0.2):
 """Hold out test_size of the rows. The test set is touched only once, at the end."""
 return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)
