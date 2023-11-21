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


# for tests, locate them to another place later
def vectorize_image(image):
    inverted_image = cv2.bitwise_not(image)
    resized_image = cv2.resize(inverted_image, DESIRED_SIZE)
    vectorized_image = resized_image.flatten()

    vectorized_image = np.asarray(vectorized_image)
    vectorized_image = vectorized_image.reshape(DESIRED_SIZE)

    # plt.imshow(vectorized_image, cmap='gray')
    # plt.axis('off')
    # plt.show()

    vectorized_image = vectorized_image.astype('float32') / 255

    return vectorized_image


# image_list must contain already vectorized images!!!
def get_test_images(image_list):
    images = [vectorize_image(image) for image in image_list]
    images = np.asarray(images)
    images = images.reshape((-1, *DESIRED_SIZE, 1))

    return images
