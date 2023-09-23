import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np

from constants import ENGLISH_DATASET, ENGLISH_DATASET_IMAGES, DESIRED_SIZE


def dataset_to_training_sample():
    # create panda dataframe from csv
    df = pd.read_csv(ENGLISH_DATASET)

    image_column = df['image'].astype(str)
    label_column = df['label'].astype(str)

    x = []
    for image in image_column:
        x.append(vectorize_image(ENGLISH_DATASET_IMAGES + image))
    y = label_column
    print(x[0])
    return x, y


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
    image_array = np.array(vectorized_image).reshape(28, 28)

    plt.imshow(image_array, cmap='gray')
    plt.axis('off')
    plt.show()


