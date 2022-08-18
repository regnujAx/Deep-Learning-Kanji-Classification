# from curses import init_pair
import data_loading, train, test

import matplotlib.pyplot as plt
import numpy as np

from keras.losses import categorical_crossentropy
from keras.optimizers import Adam

from tensorflow import keras


csv_file = "..\\chinese-mnist\\chinese_mnist.csv"
data_dir = "..\\chinese-mnist\\data\\data\\"
model_path = "model_Aug-17-2022-18-04-59\\"
model_path2 = "model_Aug-17-2022-18-51-42\\"

balancing = False
batch_size = 128
epochs = 50
learning_rate = 1e-4
upsampling_size = 25

data_loader = data_loading.DataLoader(data_dir, balancing=False, save_to_npz=False, csv_file=csv_file)

input_shape = data_loader.input_shape
# input_shape = (64,64,3)
num_classes = data_loader.num_classes
train_images = data_loader.train_images
train_labels = data_loader.train_labels
validation_images = data_loader.validation_images
validation_labels = data_loader.validation_labels
test_images = data_loader.test_images
test_labels = data_loader.test_labels
print("input_shape:", input_shape)
print("num_classes:", num_classes)
print("train_images:", train_images.shape)
print("train_labels:", train_labels.shape)
print("validation_images:", validation_images.shape)
print("validation_labels:", validation_labels.shape)
print("test_images:", test_images.shape)
print("test_labels:", test_labels.shape)


# load a model
loaded_model = keras.models.load_model(model_path2)

loaded_model.summary()

# loaded_model.trainable = False

# inputs = keras.Input(shape=input_shape)
# # We make sure that the base_model is running in inference mode here,
# # by passing `training=False`. This is important for fine-tuning, as you will
# # learn in a few paragraphs.
# x = loaded_model(inputs, training=False)
# # Convert features of shape `base_model.output_shape[1:]` to vectors
# #x = keras.layers.GlobalAveragePooling2D()(x)
# # A Dense classifier with a single unit (binary classification)
# outputs = keras.layers.Dense(num_classes)(x)
# model = keras.Model(inputs, outputs)
# model.summary()
# model.compile(optimizer=Adam(learning_rate),
#                       loss=categorical_crossentropy, 
#                       metrics=['accuracy'])

# history = model.fit(
#   train_images, train_labels, 
#   epochs=epochs, 
#   validation_data=(validation_images, validation_labels), 
#   batch_size=batch_size
# )

# train.save_model(model)
# test.plot_accuracy_and_loss(history, 0)
# test.evaluate_model(model, test_images, test_labels, batch_size)


# data = pd.read_csv("..\\chinese-mnist\\chinese_mnist.csv")

# data.head

# all_imgs = []
# for i, row in data.iterrows():
#     key = os.path.join(path, f"input_{row['suite_id']}_{row['sample_id']}_{row['code']}.jpg")
#     img = cv2.imread(key, 0)
#     # Concatenate the image with its label
#     all_imgs.append(np.r_[img.flatten(), row['code']])

# image_df = pd.DataFrame(all_imgs)

# images = image_df.drop([4096], axis=1)
# labels = image_df[4096]

# train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1, random_state=1)
# train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.11, random_state=1)

# # Dimensions of the datasets
# print(f"""
# Train set dimensions: {train_images.shape}
# Validation set dimensions: {validation_images.shape}
# Test set dimensions: {test_images.shape}
# """)


