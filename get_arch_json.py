import tensorflow as tf
from tensorflow.keras.models import load_model
import json

model = load_model('linear_regression_model.h5')

model_json = model.to_json()

with open('arch.json', 'w') as json_file:
    json_file.write(model_json)

