import tensorflow as tf
from keras.models import load_model
model = load_model("/Users/kaustubh/Desktop/EmotionGUI-UoA/models/saves/bi-lstm/mValence.hdf5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()

with open('mValence.tflite', 'wb') as f:
  f.write(tflite_model)