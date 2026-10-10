# trying the prediction run python try_predict.py
from src.predict import predict_value

sample = {"MedInc": 2.0, "HouseAge": 41, "AveRooms": 6.984, "AveBedrms": 1.0238,
 "Population": 322, "AveOccup": 2.5556, "Latitude": 37.88, "Longitude": -122.23}
print(predict_value(sample))