# this script is needed for the dataset to train a yolo model from the mnist dataset
import pandas as pd
import cv2
import numpy as np
from constants import ENGLISH_DATASET_TRAIN, ENGLISH_DATASET_TEST

train_df = pd.read_csv(ENGLISH_DATASET_TRAIN, header=None)
test_df = pd.read_csv(ENGLISH_DATASET_TEST, header=None)

df = pd.concat([train_df, test_df], ignore_index=True)


def flip_and_rotate(element):
    element = element.reshape(28, 28)
    element = np.fliplr(element)
    element = np.rot90(element)
    return element


def find_bounding_box(image):
    height, width = image.shape
    top, bottom, left, right = 0, 0, 0, 0

    for y in range(height):
        for x in range(width):
            if image[y, x] < 255:
                bottom = x, y
    for y in reversed(range(height)):
        for x in range(width):
            if image[y, x] < 255:
                top = x, y
    for x in range(width):
        for y in range(height):
            if image[y, x] < 255:
                right = x, y
    for x in reversed(range(width)):
        for y in range(height):
            if image[y, x] < 255:
                left = x, y

    new_width = abs(right[0] - left[0])
    new_height = abs(top[1] - bottom[1])
    x_c = left[0] + new_width / 2
    y_c = top[1] + new_height / 2

    new_width /= width
    new_height /= height
    x_c /= width
    y_c /= height

    return x_c, y_c, new_width, new_height


x = df.loc[:, 1:]
x = np.asarray(x)
x = np.apply_along_axis(flip_and_rotate, 1, x)
x = abs(x - 255)

y = df.loc[:, 0]
y = y.astype(int)

i = 1
for x, y in zip(x, y):
    cv2.imwrite(f'D:/datasets/i/image_{i}.png', x)
    data = find_bounding_box(x)
    with open(f'D:/datasets/t/image_{i}.txt', 'w') as file:
        file.write(f'0 {data[0]} {data[1]} {data[2]} {data[3]}')
    i += 1
