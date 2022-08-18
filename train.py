# internal script
import cnn_model

from datetime import datetime


def train_model(data_loader, batch_size, epochs, learning_rate):
    input_shape = data_loader.input_shape
    num_classes = data_loader.num_classes
    train_images = data_loader.train_images
    train_labels = data_loader.train_labels
    validation_images = data_loader.validation_images
    validation_labels = data_loader.validation_labels

    model = cnn_model.CNNModel(input_shape, num_classes, learning_rate)
    model = model.model

    model.summary()

    history = model.fit(
      train_images, train_labels, 
      epochs=epochs, 
      validation_data=(validation_images, validation_labels), 
      batch_size=batch_size
    )

    save_model(model)
    
    return model, history


def save_model(model, save_name=""):
    # Get the current date and time
    today = datetime.now()
    currentDateTime = today.strftime("%b-%d-%Y-%H-%M-%S")

    # Save the model
    save_dir = "model_" + currentDateTime
    if save_name != "":
        save_dir = save_name
    model.save(save_dir)

