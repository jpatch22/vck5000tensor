import os

def main():
    print("quantized model size: ", os.stat('quantized_model.h5').st_size)
    print("float model size: ", os.stat('linear_regression_model.h5').st_size)

if __name__ == "__main__":
    main()
