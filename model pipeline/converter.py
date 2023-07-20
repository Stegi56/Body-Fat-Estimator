import tensorflow as tf

def convert_h5_to_tflite(h5_model_path, tflite_model_path):
    # Load the .h5 model
    model = tf.keras.models.load_model(h5_model_path)

    # Convert the model to a TFLite model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the TFLite model to the specified path
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Conversion completed. TFLite model saved to '{tflite_model_path}'.")

if __name__ == "__main__":
    h5_model_path = "D:\ComputerScience\GitHub\Body-Fat-Estimator\BodyFatEstimator.h5"
    tflite_model_path = "D:\ComputerScience\GitHub\Body-Fat-Estimator\BodyFatEstimator.tflite"
    convert_h5_to_tflite(h5_model_path, tflite_model_path)