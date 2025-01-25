import os
import cv2
import time
import pandas as pd
from PIL import Image

images = []
for folder in os.listdir('../Resources/datasets/train'):
    for file in os.listdir('../Resources/datasets/train/' + folder):
        filename, extension = os.path.splitext(file)
        im = cv2.imread('../Resources/datasets/train/' + folder + "/" + filename + extension)
        time.sleep(0.1)
        images.append([im])

clusterization = pd.DataFrame(data=images, columns=['images'])
print(clusterization)
