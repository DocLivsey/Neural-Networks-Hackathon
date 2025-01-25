from keras.src.models import Model
import keras

model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=True)
model.summary()

feat_extractor = Model(inputs=model.input, outputs=model.get_layer("fc2").output)
feat_extractor.summary()
