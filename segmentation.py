import cv2
import numpy as np
import matplotlib.pyplot as plt


# calculating the intensity of dark pixels
def get_dark_pixels_intensity(edges):
    y = []
    n = edges.shape[0]
    for i in range(n):
        dark_pixels = 0
        for edge in edges:
            dark_pixels += edge[i]
        y.append(dark_pixels)

    x = np.arange(0, n)

    # plt.plot(x, y)
    # plt.show()

    return x, y


# get local minimums with step h
def get_local_minimums(x, y, h):
    minimum_x = []
    minimum_y = []
    minimum = 0
    raising = False
    for i in range(0, len(x), h):
        if y[i] <= y[minimum]:
            minimum = i
            raising = False
        elif raising:
            minimum = i
        else:
            minimum_x.append(x[minimum])
            minimum_y.append(y[minimum])
            raising = True

    # plt.plot(x, y)
    # plt.scatter(minimum_x, minimum_y, color='red')
    # plt.show()

    return minimum_x


# get filtered local minimums
# 1.) they are below the horizontal line with the y = half
# 2.) only one locates in the d area
def get_filtered_local_minimums(edges, minimums, half, d):
    below_x = []
    was = False
    for minimum in minimums:
        for y in range(0, half):
            if edges[y][minimum] > 0:
                was = True
                break
        if not was:
            below_x.append(minimum)
        was = False

    lines = []
    was = False
    n = len(below_x)
    for i in range(n):
        if was:
            was = False
            continue
        elif i == n - 1:
            lines.append(below_x[i])
        elif abs(below_x[i] - below_x[i + 1]) <= d:
            lines.append(below_x[i] + abs(below_x[i] - below_x[i + 1]) / 2)
            was = True
        else:
            lines.append(below_x[i])

    # plt.imshow(edges, cmap='gray')
    # plt.vlines(lines, ymin=0, ymax=400, colors='red', linestyles='dashed')
    # plt.show()

    return lines


# finding the bounding box of character to crop image
def find_bounding_box(image):
    height, width = image.shape
    top, bottom, left, right = 0, 0, 0, 0

    for y in range(height):
        for x in range(width):
            if image[y, x] < 128:
                bottom = y
    for y in reversed(range(height)):
        for x in range(width):
            if image[y, x] < 128:
                top = y
    for x in range(width):
        for y in range(height):
            if image[y, x] < 128:
                right = x
    for x in reversed(range(width)):
        for y in range(height):
            if image[y, x] < 128:
                left = x

    return image[top:bottom, left:right]


# adjusting the image to fit a square with white borders with centering the character
def get_square_white_image(image, offset):
    max_size = max(image.shape)
    square_image = 255 * np.ones((max_size + 2 * offset, max_size + 2 * offset), dtype=np.uint8)

    x_offset = (square_image.shape[1] - image.shape[1]) // 2
    y_offset = (square_image.shape[0] - image.shape[0]) // 2

    square_image[y_offset:y_offset + image.shape[0], x_offset:x_offset + image.shape[1]] = image

    return square_image


def word_segmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (400, 400))
    edges = cv2.Canny(image, 0, 255)
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    x, y = get_dark_pixels_intensity(edges)
    minimums = get_local_minimums(x, y, 5)
    lines = get_filtered_local_minimums(edges, minimums, 195, 15)

    plt.plot(x, y)
    plt.vlines(lines, ymin=0, ymax=16000, colors='red', linestyles='dashed')
    plt.show()
    plt.imshow(edges, cmap='gray')
    plt.vlines(lines, ymin=0, ymax=400, colors='red', linestyles='dashed')
    plt.show()

    characters = []
    n = len(lines)
    for i in range(n):
        if i == n - 1:
            cropped = image[:, lines[i]:]
        else:
            cropped = image[:, lines[i]:lines[i + 1]]
        cropped = find_bounding_box(cropped)
        character = get_square_white_image(cropped, 20)

        # plt.imshow(character, cmap='gray')
        # plt.show()

        characters.append(character)
    return characters


if __name__ == "__main__":
    word_segmentation('images/test.png')

