from keras.models import load_model
import random
import matplotlib.pyplot as plt

from constants import ENGLISH_MODEL, DESIRED_SIZE
from data_formatter import dataset_to_sample


english_model = load_model(ENGLISH_MODEL)

x, y, encoded = dataset_to_sample()
x = x.reshape(-1, DESIRED_SIZE[0], DESIRED_SIZE[1], 1)

predictions = english_model.predict(x)

displayed_samples = 50
random_indexes = random.sample(range(len(x)), displayed_samples)

fig = plt.figure(figsize=(12, 8))

for i, index in enumerate(random_indexes):
    # subfigure
    sf = fig.add_subplot(5, 10, i + 1)

    sf.imshow(x[index], cmap='gray')

    predicted_label = encoded[predictions[index].argmax()]
    true_label = encoded[predictions[index].argmax()]

    sf.set_title(f'Predicted: {predicted_label}\nReality: {true_label}')
    sf.axis("off")

plt.tight_layout()
plt.show()
