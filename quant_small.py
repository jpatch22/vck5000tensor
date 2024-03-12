import tensorflow as tf
import os
from tensorflow_model_optimization.quantization.keras import vitis_quantize
from PIL import Image
import numpy as np
import pandas as pd
import random


def files_in_folder(folder_path):
    return os.listdir(folder_path)

def main():
    float_model = tf.keras.models.load_model("linear_regression_model.h5")
    print("Summary of float model")
    print(float_model.summary())
    print("END FLOAT MODEL")
    data = np.load("training_data.npz")
    calib_data = [data["X_train"], data["y_train"]]
    #print(data, data["X_train"])
    print(calib_data)
    
    quantizer = vitis_quantize.VitisQuantizer(float_model)
    print("HERE")
    print("Quant", quantizer)
    
    quantized_model = quantizer.quantize_model(calib_dataset=data["X_train"], 
            calib_steps=100, 
            calib_batch_size=10,
            add_shape_info=False)
            #input_shape=[None, 64, 64, 1])
    print("Summary of quantized model")
    print(quantized_model.summary())
    print("END QUANTIZED MODEL")
    quantized_model.save("quantized_model.h5")

if __name__ == "__main__":
    main()

