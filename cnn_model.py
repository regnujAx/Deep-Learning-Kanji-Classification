from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam


class CNNModel():
    def __init__(self, input_shape, num_classes, learning_rate):
        self.model = self.initialize_model(input_shape, num_classes, learning_rate)
    
    def initialize_model(self, input_shape, num_classes, learning_rate):
        # Define a CNN model
        model = Sequential()

        model.add(Conv2D(32, (5, 5), strides=(1, 1), activation="relu", input_shape=input_shape, padding="same"))

        model.add(Conv2D(32, (5, 5), strides=(2, 2), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same"))

        model.add(Conv2D(64, (3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Conv2D(96, (3, 3), strides=(1, 1), activation="relu", padding="same"))

        model.add(Conv2D(96, (3, 3), strides=(1, 1), activation="relu", padding="same"))
        model.add(MaxPooling2D((2, 2)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))

        model.add(Flatten())
        model.add(Dense(128, activation="relu"))
        model.add(Dense(num_classes, activation="softmax", kernel_regularizer="l2"))

        model.compile(
            optimizer=Adam(learning_rate),
            loss=categorical_crossentropy,
            metrics=["accuracy"])

        return model
