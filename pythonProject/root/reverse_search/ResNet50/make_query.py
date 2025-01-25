from root.reverse_search.ResNet50.all_paths import DATASET_PATH, PATH_TO_QUERY_FOLDER, TEST_IMAGES, FEATURES
from root.reverse_search.helpful_functions import get_features, show_images
from root.reverse_search.ResNet50.load_model import model
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
from PIL import Image
import pickle as pk
import numpy as np
import os


query_file = os.listdir(PATH_TO_QUERY_FOLDER)
if len(query_file) == 1:
    print("len is ok")
    if query_file[0].endswith('.jpg') or query_file[0].endswith('.png'):
        print("extension is ok")
        query_image_pillow = Image.open(f'{PATH_TO_QUERY_FOLDER}\\{query_file[0]}').convert('RGB')
        query_image_features = get_features(query_image_pillow, model)
        plt.imshow(query_image_pillow)
        plt.title('Query Image')
        plt.show()
        print(query_image_features.shape)

        image_features = pk.load(open(f"{FEATURES}", "rb"))
        features = []
        for image in image_features:
            features.append(np.array(image['features']))
        features = np.array(features)
        features = np.squeeze(features)

        knn = NearestNeighbors(n_neighbors=20, algorithm='kd_tree', metric='l2')
        knn.fit(features)
        file_names = os.listdir(DATASET_PATH)

        indices = knn.kneighbors([query_image_features], return_distance=False)
        found_images = []
        for x in indices[0]:
            image = Image.open(DATASET_PATH + "/" + file_names[x])
            image = image.resize((224, 224))
            found_images.append(np.array(image))

        found_images = np.array(found_images)
        #found_images = found_images.reshape((20, 224, 224, 3))
        show_images(found_images)
    else:
        raise RuntimeError(f'there is no file with .jpg or .png extensions in {PATH_TO_QUERY_FOLDER}')
else:
    raise RuntimeError(f'there should be only one file in the folder with the extension .jpg or .png'
                       f' in {PATH_TO_QUERY_FOLDER}')
