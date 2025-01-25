from root.reverse_search.VGG16.model import model, feat_extractor
from root.reverse_search.helpful_functions import load_image, image2array
from sklearn.decomposition import PCA
import numpy as np
import time


PATH = '../../Resources/training'
images = image2array(path=PATH)
tic = time.perf_counter()
features = []
for i, image_path in enumerate(PATH[:200]):
    if i % 500 == 0:
        toc = time.perf_counter()
        elap = toc-tic
        print("analyzing image %d / %d. Time: %4.4f seconds." % (i, len(images), elap))
        tic = time.perf_counter()
    img, x = load_image(PATH, model)
    feat = feat_extractor.predict(x)[0]
    features.append(feat)
print('finished extracting features for %d images' % len(images))

features = np.array(features)
pca = PCA(n_components=100)
pca.fit(features)

pca_features = pca.transform(features)
