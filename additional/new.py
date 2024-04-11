import tensorflow as tf

# Загрузка модели Keras
model = tf.keras.models.load_model('./models/english_model.h5')

# Конвертация модели в формат TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Сохранение модели в файл .tflite
with open('./models/english_model.tflite', 'wb') as f:
    f.write(tflite_model)
