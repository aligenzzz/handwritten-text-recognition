from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model
from keras import layers
from keras.src.utils import np_utils
import tensorflow as tf

from data_formatter import dataset_to_sample

x, y, encoded = dataset_to_sample()

number_of_classes = y.nunique()
y = np_utils.to_categorical(y, number_of_classes)

x = x.reshape(-1, 28, 28, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=88)

english_model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPool2D(strides=2),
    layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'),
    layers.MaxPool2D(strides=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(number_of_classes, activation='softmax')
])
# english_model.summary()

english_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = english_model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=1,
                            validation_data=(x_test, y_test))
tf.saved_model.save(english_model, 'models')


