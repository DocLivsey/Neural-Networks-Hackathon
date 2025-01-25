from root.reverse_search.ResNet50.all_paths import FEATURES
from keras.api.preprocessing import image
from keras.src.applications.convnext import preprocess_input
from keras.src.models import model
import matplotlib.pyplot as plt
from os.path import splitext
from pathlib import Path
from skimage import io
from os import listdir
from tqdm import tqdm
from PIL import Image
import pickle as pk
import numpy as np
import cv2
import os

PATH = '../Resources/training'


def is_file_in_folder(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    return os.path.exists(file_path)


def image2array(filelist=None, path=''):
    image_array = []
    if filelist is None:
        filelist = os.listdir(path)
    for image in filelist:
        img = io.imread(path + '/' + image)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        image_array.append(img)
    image_array = np.array(image_array)
    image_array = image_array.reshape(image_array.shape[0], 224, 224, 3)
    image_array = image_array.astype('float32')
    image_array /= 255
    return np.array(image_array)


'''
train_data = image2array(path=PATH)
print("Length of training dataset:", train_data.shape)
'''


def load_image(path, _model):
    print(_model.input_shape)
    img = image.load_img(path, target_size=_model.input_shape[1:])
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return img, x


def show_images(images, figsize=(20, 10), columns=5):
    plt.figure(figsize=figsize)
    for i, img in enumerate(images):
        plt.subplot(int(len(images) / columns + 1), columns, i + 1)
        plt.imshow(img)
    plt.show()


def read_img_file(f):
    img = Image.open(f)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


def resize_img_to_array(img, img_shape):
    img_array = np.array(
        img.resize(
            img_shape,
            Image.Resampling.NEAREST
        )
    )
    return img_array


def get_features(img, _model):
    img_width, img_height = 224, 224
    np_img = resize_img_to_array(img, img_shape=(img_width, img_height))
    expanded_img_array = np.expand_dims(np_img, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    x_conv = _model.predict(preprocessed_img)
    image_features = x_conv[0]
    image_features /= np.linalg.norm(image_features)
    return image_features


def generate_resnet_features(path_to_files_folder, _model):
    all_image_features = []
    image_filenames = listdir(path_to_files_folder)
    image_ids = set(map(lambda el: splitext(el)[0], image_filenames))
    try:
        all_image_features = pk.load(open(f"{FEATURES}", "rb"))
    except (OSError, IOError) as e:
        print("file_not_found")

    def exists_in_all_image_features(image_id):
        for image in all_image_features:
            if image['image_id'] == image_id:
                return True
        return False

    def exists_in_image_folder(image_id):
        if image_id in image_ids:
            return True
        return False

    def sync_resnet_image_features():
        for_deletion = []
        for i in range(len(all_image_features)):
            if not exists_in_image_folder(all_image_features[i]['image_id']):
                print("deleting " + str(all_image_features[i]['image_id']))
                for_deletion.append(i)
        for i in reversed(for_deletion):
            del all_image_features[i]

    sync_resnet_image_features()
    for image_filename in tqdm(image_filenames):
        image_id = splitext(image_filename)[0]
        if exists_in_all_image_features(image_id):
            continue
        img_arr = read_img_file(path_to_files_folder + "/" + image_filename)
        image_features = get_features(img_arr, _model)
        all_image_features.append({'image_id': image_id, 'features': image_features})
    pk.dump(all_image_features, open(f"{FEATURES}", "wb"))
