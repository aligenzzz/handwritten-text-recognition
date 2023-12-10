import cv2
import numpy as np
import matplotlib.pyplot as plt
from yolo import get_yolo_boxes


# dividing into separate lines
def get_lines(image):
    height, width = image.shape
    lines = []
    was = False
    for line, j in zip(image, range(height)):
        dark = False
        for i in range(width):
            if line[i] < 128:
                dark = True
            if i == width - 1 and line[i] >= 128 and not was and not dark:
                was = True
            if line[i] < 128 and was:
                lines.append(j)
                was = False
                break

    # plt.imshow(image)
    # plt.hlines(lines, xmin=0, xmax=width, colors='red', linestyles='dashed')
    # plt.show()

    n = len(lines)
    if n == 0:
        return [image]

    result_lines = []
    for i in range(-1, n):
        if i == n - 1:
            cropped = image[lines[i]:, :]
        elif i == -1:
            cropped = image[:lines[i + 1], :]
        else:
            cropped = image[lines[i]:lines[i + 1], :]
        cropped = find_bounding_box(cropped)

        # plt.imshow(cropped, cmap='gray')
        # plt.show()

        result_lines.append(cropped)

    return result_lines


# dividing line into words
def get_words(line):
    height, width = line.shape
    words = []
    was = False
    for i in range(width):
        dark = False
        for j in range(height):
            if line[j, i] < 128:
                dark = True
            if j == height - 1 and line[j, i] >= 128 and not was and not dark:
                was = True
                words.append(i)
            if line[j, i] < 128 and was:
                words.append(i)
                was = False
                break

    copy_words = words.copy()
    n = len(copy_words)
    for i in range(0, n, 2):
        if i == n - 1:
            continue
        if abs(copy_words[i] - copy_words[i + 1]) < 40:
            words.remove(copy_words[i])
            words.remove(copy_words[i + 1])
            pass
        else:
            words.remove(copy_words[i])

    # plt.imshow(line)
    # plt.vlines(words, ymin=0, ymax=height, colors='red', linestyles='dashed')
    # plt.show()

    if len(words) == 0:
        return [line]

    result_words = []
    n = len(words)
    for i in range(-1, n):
        if i == n - 1:
            cropped = line[:, words[i]:]
        elif i == -1:
            cropped = line[:, :words[i + 1]]
        else:
            cropped = line[:, words[i]:words[i + 1]]
        cropped = find_bounding_box(cropped)

        # plt.imshow(cropped, cmap='gray')
        # plt.show()

        result_words.append(cropped)

    return result_words


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


# enlarging image
def enlarge(image, scale_factor):
    new_width = int(image.shape[1] * scale_factor)
    new_height = int(image.shape[0] * scale_factor)
    new_size = (new_width, new_height)

    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

    return resized_image


def word_segmentation(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = find_bounding_box(image)

    lines = get_lines(image)
    text = []
    for line in lines:
        text.append(get_words(line))

    words = []
    for i in range(len(text)):
        for j in range(len(text[i])):
            new_image, start_boxes = get_yolo_boxes(cv2.cvtColor(get_square_white_image(enlarge(text[i][j], 2), 20),
                                                                 cv2.COLOR_GRAY2BGR))
            boxes = sorted(start_boxes, key=lambda x: x[0])
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

            characters = []
            for box in boxes:
                cropped = new_image[round(box[2]):round(box[3]), round(box[0]):round(box[1])]
                cropped = find_bounding_box(cropped)
                character = get_square_white_image(cropped, 10)

                # plt.imshow(character, cmap='gray')
                # plt.show()

                characters.append(character)

            words.append(characters)

    return words

