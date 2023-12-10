import numpy as np
import matplotlib.pyplot as plt


# calculating the intensity of dark pixels
def get_dark_pixels_intensity(edges):
    y = []
    n = edges.shape[1]
    for i in range(n):
        dark_pixels = 0
        for edge in edges:
            dark_pixels += edge[i]
        y.append(dark_pixels)

    x = np.arange(0, n)

    plt.plot(x, y)
    plt.show()

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

    plt.plot(x, y)
    plt.scatter(minimum_x, minimum_y, color='red')
    plt.show()

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

    plt.imshow(edges, cmap='gray')
    plt.vlines(lines, ymin=0, ymax=edges.shape[0], colors='red', linestyles='dashed')
    plt.show()

    return lines
