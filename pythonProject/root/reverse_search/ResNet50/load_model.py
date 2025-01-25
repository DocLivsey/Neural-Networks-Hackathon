from keras.src.applications.resnet import ResNet50
from root.reverse_search.ResNet50.all_paths import DATASET_PATH, FEATURES
from root.reverse_search.helpful_functions import generate_resnet_features
import os


model = ResNet50(weights='imagenet', include_top=False,
                 input_shape=(224, 224, 3), pooling='max')

if not os.path.exists(FEATURES):
    generate_resnet_features(DATASET_PATH, model)
