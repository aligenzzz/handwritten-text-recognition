from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras import layers
from keras.src.utils import np_utils
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from data_formatter import dataset_to_sample
from constants import DESIRED_SIZE, ENGLISH_MODEL, GRAPHICS

x, y, encoded = dataset_to_sample()

number_of_classes = y.nunique()
y = np_utils.to_categorical(y, number_of_classes)

A, B = DESIRED_SIZE

# tensorflow requires 4D Array
x = x.reshape(-1, A, B, 1)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, stratify=y, random_state=88)

english_model = Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(A, B, 1)),
    layers.MaxPool2D(strides=2),
    layers.Conv2D(filters=48, kernel_size=(5, 5), padding='valid', activation='relu'),
    layers.MaxPool2D(strides=2),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(84, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(number_of_classes, activation='softmax')
])
english_model.summary()

english_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# if it doesn't improve for 5 epochs
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min')
# save the best model during training based on validation loss
checkpoint = ModelCheckpoint(ENGLISH_MODEL, save_best_only=True, monitor='val_loss', verbose=1, mode='auto')
# reduce the learning rate if validation loss plateaus for 3 epochs
RLP = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.2, min_lr=0.0001)

english_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = english_model.fit(x_train, y_train, epochs=30, batch_size=32, verbose=1,
                            validation_data=(x_val, y_val),
                            callbacks=[early_stopping, checkpoint, RLP])

# plot graphics
plt.title('Losses train/validation')
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.legend()
plt.savefig(f'{GRAPHICS}em_losses.png')
plt.show()

plt.title('Accuracies train/validation')
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.legend()
plt.savefig(f'{GRAPHICS}em_accuracies.png')
plt.show()
