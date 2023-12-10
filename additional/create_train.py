# this script is needed to create a train.txt file that stores the paths to all train images
import os

path = 'data/obj/'
image_list = os.listdir('../../../ЗАГРУЗКИ/project-2-at-2023-11-26-15-39-03852328/images')
print(image_list)

file = open('../../../ЗАГРУЗКИ/project-2-at-2023-11-26-15-39-03852328/train.txt', 'w')

for image in image_list:
    imagePath = path + image + '\n'
    file.write(imagePath)

file.close()
