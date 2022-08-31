# internal script
import cnn_model

import time

from datetime import datetime


def train_model(data_loader, batch_size, epochs, learning_rate, label):
    input_shape = data_loader.input_shape
    num_classes = data_loader.num_classes
    train_images = data_loader.train_images
    train_labels = data_loader.train_labels
    validation_images = data_loader.validation_images
    validation_labels = data_loader.validation_labels

    model = cnn_model.CNNModel(input_shape, num_classes, learning_rate)
    model = model.model

    model.summary()

    start_time = time.time()

    history = model.fit(
      train_images, train_labels,
      epochs=epochs,
      validation_data=(validation_images, validation_labels),
      batch_size=batch_size)

    elapsed_time = time.time() - start_time
    time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print("\nElapsed time for training:", time_string)

    save_model(model, label)

    return model, history


def save_model(model, label):
    # Get the current date and time
    today = datetime.now()
    currentDateTime = today.strftime("%b-%d-%Y-%H-%M-%S")

    # Save the model
    save_file_name = f"model_{currentDateTime}"
    if label != "":
        save_file_name = f"{label}_{save_file_name}"
    model.save(f"{save_file_name}.h5")
