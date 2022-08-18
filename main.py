# internal scripts
import data_loading, test, train, transfer_learning

import argparse
import keras


### command-line parser ###
parser = argparse.ArgumentParser(description="Train or run a neural network model for image classification.")
parser.add_argument("-b", "--balancing", choices=("True","False"), required=False, help="Balance the data")
parser.add_argument("-bs", "--batch_size", type=int, required=False, help="Batch size")
parser.add_argument("-c", "--csv_file", type=str, required=False, help="Path to a csv file for data loading")
parser.add_argument("-d", "--data", required=True, help="Path to the dataset to train the model")
parser.add_argument("-e", "--epochs", type=int, required=False, help="Number of training epochs")
parser.add_argument("-l", "--label", type=str, required=False, help="Add a individual label")
parser.add_argument("-ln", "--load_from_npz", choices=("True","False"), required=False, help="Load data from npz files (only with -np/--npz_paths argument to load from the passed paths)")
parser.add_argument("-lr", "--learning_rate", type=float, required=False, help="Learning rate")
parser.add_argument("-mp", "--model_path", type=str, required=False, help="Path to a local pretrained model to load it")
parser.add_argument("-np", "--npz_paths", nargs="+", required=False, help="Paths to the npz files that contains the train images, train labels, test images and test labels "
                                                                        + " (only with -ln/--load_from_npz=\"True\") "
                                                                        + "use e.g. \'-np path/to/train_images.npz path/to/train_labels.npz path/to/test_images.npz path/to/test_labels.npz\'")
parser.add_argument("-pm", "--pretrained_model", choices=['resnet50', 'vgg16'], required=False, help="Select a pretrained model for transfer learning")
parser.add_argument("-sn", "--save_to_npz", choices=("True","False"), required=False, help="Save the loaded data as npz files")
parser.add_argument("-us", "--upsampling_size", type=int, required=False, help="Upsampling size")
parser.add_argument("-g", "--use_grayscale", choices=("True","False"), required=False, help="If using grayscale images (8 bit images) for pretrained model that was trained on rgb images (24 bit images)")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    exit(0)

balancing = True
batch_size = 128
csv_file = ""
epochs = 50
label = ""
learning_rate = 1e-4
load_from_npz = False
# npz_paths = ["train_images.npz", "train_labels.npz", "test_images.npz", "test_labels.npz"]
npz_paths = []
model_path = ""
save_to_npz = True
upsampling_size = 25
use_grayscale = False

data_dir = args.data

if args.balancing == "False":
    balancing = False

if args.batch_size:
    batch_size = args.batch_size

if args.csv_file:
    csv_file = args.csv_file

if args.epochs:
    epochs = args.epochs

if args.label:
    label = args.label

if args.learning_rate:
    learning_rate = args.learning_rate

if args.load_from_npz == "True":
    load_from_npz = True
    for path in args.npz_paths:
        npz_paths.append(path)

if args.save_to_npz == "False":
    save_to_npz = False
    
if args.upsampling_size:
    upsampling_size = args.upsampling_size

if args.use_grayscale == "True":
    use_grayscale = True


try:
    data_loader = data_loading.DataLoader(
        data_dir,
        balancing=balancing, upsampling_size=upsampling_size,
        load_from_npz=load_from_npz, npz_paths=npz_paths,
        save_to_npz=save_to_npz,
        csv_file=csv_file,
        use_grayscale=use_grayscale)

    test_images = data_loader.test_images
    test_labels = data_loader.test_labels

    if args.model_path:
        model = keras.models.load_model(args.model_path)
        transfer_learning.transfer_learning_with_local_model(model, data_loader, batch_size, epochs, learning_rate, label=label)
    elif args.pretrained_model:
        model = keras.models.load_model(args.model_path)
        transfer_learning.transfer_learning_with_pretrained_model(model, data_loader, batch_size, epochs, learning_rate)
    else:
        model, history = train.train_model(data_loader, batch_size, epochs, learning_rate)
        test.plot_accuracy_and_loss(history, label=label)
        test.evaluate_model(model, test_images, test_labels, batch_size)
except Exception as e:
    print(e)
