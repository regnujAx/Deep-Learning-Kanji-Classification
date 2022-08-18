import data_loading, train#, test

import numpy as np

import keras.applications.resnet_v2

# from tensorflow import keras
import keras.applications.inception_v3
# from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D



csv_file = "..\\chinese-mnist\\chinese_mnist.csv"
csv_file_short = "..\\chinese-mnist\\chinese_mnist_short.csv"
data_dir = "..\\chinese-mnist\\data\\data\\"
model_path = "..\\model_Aug-10-2022-15-27-36\\"
model_name = "resnet50"

balancing = False
batch_size = 128
epochs = 50
learning_rate = 1e-4
# npz_paths = ["train_images.npz", "train_labels.npz", "test_images.npz", "test_labels.npz"]
npz_paths = []

data_loader = data_loading.DataLoader(data_dir, balancing=balancing, csv_file=csv_file, use_grayscale=True)

input_shape = data_loader.input_shape
print("input_shape:", input_shape)
# input_shape = (64,64,3)
# num_classes = data_loader.num_classes
train_images = data_loader.train_images
print(train_images.shape)
# train_labels = data_loader.train_labels
# validation_images = data_loader.validation_images
# validation_labels = data_loader.validation_labels

# print(train_images.shape)  # (64, 224, 224)    

# rgb_batch = np.repeat(train_images[..., np.newaxis], 3, -1)    
# print(rgb_batch.shape)  # (64, 224, 224, 3)



# x = keras.applications.resnet_v2.preprocess_input(train_images)
# print(x)
# print(x.shape)

# # create the base pre-trained model
# base_model = InceptionV3(weights='imagenet', include_top=False)

# # add a global spatial average pooling layer
# x = base_model.output
# x = GlobalAveragePooling2D()(x)
# # let's add a fully-connected layer
# x = Dense(1024, activation='relu')(x)
# # and a logistic layer
# predictions = Dense(num_classes, activation='softmax')(x)

# # this is the model we will train
# model = Model(inputs=base_model.input, outputs=predictions)

# # first: train only the top layers (which were randomly initialized)
# # i.e. freeze all convolutional InceptionV3 layers
# for layer in base_model.layers:
#     layer.trainable = False

# # compile the model (should be done *after* setting layers to non-trainable)
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

# # train the model on the new data for a few epochs
# model.fit(
#   train_images, train_labels, 
#   epochs=epochs, 
#   validation_data=(validation_images, validation_labels), 
#   batch_size=batch_size
# )

# train.save_model(model)

# # at this point, the top layers are well trained and we can start fine-tuning
# # convolutional layers from inception V3. We will freeze the bottom N layers
# # and train the remaining top layers.

# # let's visualize layer names and layer indices to see how many layers
# # we should freeze:
# for i, layer in enumerate(base_model.layers):
#    print(i, layer.name)

# # we chose to train the top 2 inception blocks, i.e. we will freeze
# # the first 249 layers and unfreeze the rest:
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True

# # we need to recompile the model for these modifications to take effect
# # we use SGD with a low learning rate
# from keras.optimizers import SGD
# model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

# # we train our model again (this time fine-tuning the top 2 inception blocks
# # alongside the top Dense layers
# model.fit(
#   train_images, train_labels, 
#   epochs=epochs, 
#   validation_data=(validation_images, validation_labels), 
#   batch_size=batch_size
# )

# train.save_model(model)
