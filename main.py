# internal scripts
import data_loading, train, test, plot

import argparse

from tensorflow import keras


### command-line parser ###
parser = argparse.ArgumentParser(description="Train or run a neural network model for image classification.")
parser.add_argument("-b", "--balancing", choices=("True","False"), required=False, help="Balance the data")
parser.add_argument("-bs", "--batch_size", type=int, required=False, help="Batch size")
parser.add_argument("-d", "--data", required=True, help="Path to the dataset to train the model")
parser.add_argument("-e", "--epochs", type=int, required=False, help="Number of training epochs")
parser.add_argument("-f", "--model_path", type=str, required=False, help="Folder path to pretrained model to load it")
parser.add_argument("-l", "--load_from_npz", choices=("True","False"), required=False, help="Load data from npz files (only with -p argument to load from the passed paths)")
parser.add_argument("-lr", "--learning_rate", type=float, required=False, help="Learning rate")
parser.add_argument("-np", "--npz_paths", nargs='+', required=False, help="Paths to the npz files that contains the train images, train labels, test images and test labels "
                                                                        + "(use e.g. \'-p path/to/train_images.npz path/to/train_labels.npz path/to/test_images.npz path/to/test_labels.npz\'")
parser.add_argument("-p", "--pretrained_model", type=str, required=False, help="Path to pretrained model")
parser.add_argument("-s", "--save_to_npz", choices=("True","False"), required=False, help="Save the loaded data as npz files")
parser.add_argument("-u", "--upsampling_size", type=int, required=False, help="Upsampling size")
#parser.add_argument("-optim", "--optimizer", type=str, required=False, help="Set optimizer type")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    exit(0)

balancing = True
batch_size = 128
epochs = 50
learning_rate = 1e-4
upsampling_size = 25
load_from_npz = False
# npz_paths = ["train_images.npz", "train_labels.npz", "test_images.npz", "test_labels.npz"]
npz_paths = []

data_dir = args.data

if args.balancing == "False":
    balancing = False

if args.batch_size:
    batch_size = args.batch_size

if args.epochs:
    epochs = args.epochs

if args.learning_rate:
    learning_rate = args.learning_rate

if args.load_from_npz == "True":
    load_from_npz = True
    for path in args.npz_paths:
        npz_paths.append(path)
    
if args.upsampling_size:
    upsampling_size = args.upsampling_size


# data_dir = '..\\archive\\kkanji\\kkanji2'

# model_path = "model_Aug-17-2022-14-16-52"
# data_loader = data_loading.DataLoader(data_dir, balancing, upsampling_size)
# # load a model
# loaded_model = keras.models.load_model(model_path)
# test.evaluate_model(loaded_model, data_loader.test_images, data_loader.test_labels, batch_size)

try:
    model = keras.models.load_model("model_Aug-17-2022-18-04-59\\")

    data_loader = data_loading.DataLoader(data_dir, balancing, upsampling_size, load_from_npz, npz_paths)

    train_images = data_loader.train_images
    train_labels = data_loader.train_labels
    validation_images = data_loader.validation_images
    validation_labels = data_loader.validation_labels
    test_images = data_loader.test_images
    test_labels = data_loader.test_labels

    plot.plot_predictions(model, test_images, test_labels)
    # if args.model_path:
    #     model = keras.models.load_model(args.model_path)
    # else:
    #     if args.pretrained_model:
    #         pretrained_model = args.pretrained_model
    #     else:
    #         model, history = train.train_model(data_loader, batch_size, epochs, learning_rate)
    #         test.plot_accuracy_and_loss(history, 1)

    # test.evaluate_model(model, data_loader.test_images, data_loader.test_labels, batch_size)
    

    # data_loader2 = data_loading.DataLoader(data_dir, balancing, 50, load_from_npz)
    # model2, history2 = train.train_model(data_loader2, batch_size, epochs, learning_rate)
    # test.plot_accuracy_and_loss(history2, 2)
    # test.evaluate_model(model2, data_loader2.test_images, data_loader2.test_labels, batch_size)

    # data_loader3 = data_loading.DataLoader(data_dir, balancing, 75, load_from_npz)
    # model3, history3 = train.train_model(data_loader3, batch_size, epochs, learning_rate)
    # test.plot_accuracy_and_loss(history3, 3)
    # test.evaluate_model(model3, data_loader3.test_images, data_loader3.test_labels, batch_size)
except Exception as e:
    print(e)
