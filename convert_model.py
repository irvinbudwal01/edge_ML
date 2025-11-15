import tensorflow as tf

#keras_model = tf.keras.models.load_model("saved_env_model.keras")

#keras_model = tf.keras.models.load_model("saved_pdm_model.keras")

keras_model = tf.keras.models.load_model("saved_env_dropped_model.keras")

converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

#converter.optimizations = [tf.lite.Optimize.DEFAULT]
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.int8  # Specify input and output types for integer quantization
#converter.inference_output_type = tf.int8

tf_lite_model = converter.convert()

#with open('env_model.tflite', 'wb') as f:
#    f.write(tf_lite_model)

with open('env_dropped_model.tflite', 'wb') as f:
    f.write(tf_lite_model)
