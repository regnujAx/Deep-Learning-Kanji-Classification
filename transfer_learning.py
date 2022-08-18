# internal scripts
import test, train


from tensorflow import keras

from keras import Input
from keras.applications.resnet import ResNet50
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, GlobalAveragePooling2D
from keras.losses import categorical_crossentropy
from keras.models import Model
from keras.optimizers import Adam


### Transfer learning based on https://keras.io/guides/transfer_learning/

def transfer_learning_with_local_model(loaded_model, data_loader, batch_size, epochs, learning_rate, label="transfer_learning"):
    input_shape = data_loader.input_shape
    num_classes = data_loader.num_classes
    train_images = data_loader.train_images
    train_labels = data_loader.train_labels
    validation_images = data_loader.validation_images
    validation_labels = data_loader.validation_labels
    test_images = data_loader.test_images
    test_labels = data_loader.test_labels

    ## loaded_model.summary()

    # Freeze the loaded model
    loaded_model.trainable = False

    inputs = Input(shape=input_shape)

    # Run the loaded model in inference mode
    x = loaded_model(inputs, training=False)

    # Add a Dense layer with num_classes units
    outputs = Dense(num_classes)(x)

    model = Model(inputs, outputs)

    model.summary()

    model.compile(
        optimizer=Adam(learning_rate),
        loss=categorical_crossentropy,
        metrics=["accuracy"])

    # Train the model on the new data
    history = model.fit(
        train_images, train_labels, 
        epochs=epochs, 
        validation_data=(validation_images, validation_labels), 
        batch_size=batch_size)

    train.save_model(model, save_name=label)
    test.plot_accuracy_and_loss(history, label)
    test.evaluate_model(model, test_images, test_labels, batch_size)
    test.plot_predictions(model, test_images, test_labels, label=label)


def transfer_learning_with_pretrained_model(model_name, data_loader, batch_size, epochs, learning_rate):
    input_shape = data_loader.input_shape
    num_classes = data_loader.num_classes
    train_images = data_loader.train_images
    train_labels = data_loader.train_labels
    validation_images = data_loader.validation_images
    validation_labels = data_loader.validation_labels
    test_images = data_loader.test_images
    test_labels = data_loader.test_labels

    if model_name == "vgg16":
        # Create the base pretrained VGG16 model
        base_model = VGG16(weights="imagenet", include_top=False, input_shape=input_shape)
    else:
        # Create the base pretrained ResNet50 model
        base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape)
    
    # Add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # Add a fully-connected layer
    x = Dense(1024, activation="relu")(x)
    # Add a logistic layer
    predictions = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.summary()

    # Train only the top layers, 
    # i.e. freeze all convolutional base_model dependant layers
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"])

    # Train the model on the new data
    history = model.fit(
        train_images, train_labels,
        epochs=epochs,
        validation_data=(validation_images, validation_labels),
        batch_size=batch_size)

    train.save_model(model, save_name=model_name)
    test.plot_accuracy_and_loss(history, model_name)
    test.plot_predictions(model, test_images, test_labels, label=model_name, channels=3)
    test.evaluate_model(model, test_images, test_labels, batch_size)
