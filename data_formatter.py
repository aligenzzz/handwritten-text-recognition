import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import ENGLISH_DATASET_TRAIN, ENGLISH_DATASET_TEST, \
                      ENGLISH_DATASET_MAPPING, DESIRED_SIZE


def dataset_to_sample():
    train_df = pd.read_csv(ENGLISH_DATASET_TRAIN, header=None)
    test_df = pd.read_csv(ENGLISH_DATASET_TEST, header=None)

    df = pd.concat([train_df, test_df], ignore_index=True)

    def flip_and_rotate(element):
        element = element.reshape(DESIRED_SIZE)
        element = np.fliplr(element)
        element = np.rot90(element)
        return element

    x = df.loc[:, 1:]
    x = x.astype('float32') / 255
    x = np.asarray(x)
    x = np.apply_along_axis(flip_and_rotate, 1, x)

    y = df.loc[:, 0]
    y = y.astype(int)

    encoded = get_encoded()

    return x, y, encoded


def get_encoded():
    mapping = pd.read_csv(ENGLISH_DATASET_MAPPING, delimiter=' ', index_col=0, header=None)
    mapping = mapping.iloc[:, 0]

    encoded = dict()
    for index, label in enumerate(mapping):
        encoded[index] = chr(label)

    return encoded


def vectorize_image(image_path):
    image = cv2.imread(image_path)
    inverted_image = cv2.bitwise_not(image)
    grayscale_image = cv2.cvtColor(inverted_image, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(grayscale_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    x_min, y_min, x_max, y_max = np.inf, np.inf, -np.inf, -np.inf

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x + w)
        y_max = max(y_max, y + h)

    cropped_image = grayscale_image[y_min:y_max, x_min:x_max]
    resized_image = cv2.resize(cropped_image, DESIRED_SIZE)

    vectorized_image = resized_image.flatten()

    # plot_image(vectorized_image)

    return vectorized_image


def plot_image(vectorized_image):
    image_array = np.array(vectorized_image).reshape(DESIRED_SIZE)

    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    plt.show()


