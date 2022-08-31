# Kanji Classification

This is the practical part of a project that took place as part of the Deep Learning course at the Hasso Plattner Institute under the supervision of Prof. Dr. Lippert.<br />
The goal of this lecture was to train a model with the data of [Kuzushiji characters](https://www.kaggle.com/datasets/anokas/kuzushiji). After training, we should use the model for transfer learning on the [Chinese MNIST dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist). We were also advised to try alternatives to CNNs.

## Requirements

- Python 3.7 or higher

For the following requirements you could run ```pip -r requirements.txt``` (or something similar):
- argparse
- datetime
- keras
- matplotlib
- numpy
- os
- pandas
- PIL
- sklearn
- time

## How to use

Use -h/--help to show all possible arguments for the main.py.

You can run the main.py with different (optional) arguments:<br />
-b/--balancing: Can be True or False and indicates whether the data should be balanced or not (default is True).<br />
-bs/--batch_size: Can be an integer which indicates the batch size for training and testing (default is 128).<br />
-c/--csv_file: Can be a (relational) path to a CSV file which is necessary for data loading for the Chinese MNIST dataset (default is an empty string).<br />
-d/--data: Can be a (relational) path to a directory that contains the dataset to train the model. The directory should have the structure of subdirectories (named with the labels of the dataset) which contain the respective images (default is an empty string).<br />
-e/--epochs: Can be an integer which indicates the number of training epochs (default is 50).<br />
-g/--use_grayscale: Can be True or False and indicates whether grayscale images (8 bit images) will be used for transfer learning on a pretrained model that was trained on rgb images (24 bit images) (default is False).<br />
-l/--label: Can be a string (without quotes) to add an individual label for the saved files (default is an empty string).<br />
-ln/--load_from_npz: Can be True or False and indicates whether the data should be loaded from npz files (only with -np/--npz_paths argument to load from the passed paths) (default is False).<br />
-lr/--learning_rate: Can be a float which indicates the learning rate (default is 0.0001).<br />
-mp/--model_path: Can be a (relational) path to a local pretrained model to load it for transfer learning.<br />
-np/--npz_paths: Can be a list of four (relational) paths to the npz files that contains the train images, train labels, test images and test labels (only with -ln True). For an example command see below (default is an empty list).<br />
-pm/--pretrained_model: Can be resnet50 or vgg16 for selecting a pretrained model (ResNet50 or VGG16) for transfer learning.<br />
-rs/--resize_shape: Can be two numbers which indicate the shape (width and height) of images to be resized (e.g., use \'-rs 224 224\' for the shape (224,224)) (default is (64,64)).<br />
-sn/--save_to_npz: Can be True or False and indicates whether the loaded data should be saved as npz files (default is True).<br />
-us/--upsampling_size: Can be an integer which indicates the upsampling size (default is 25).

**HINT: Because of the defaults of the most arguments make sure that you pass the correct values.**

Example commands:
- You can run the following command if you have downloaded and extracted the [Kuzushiji dataset](https://www.kaggle.com/datasets/anokas/kuzushiji) and want to use the default values (here with the label \'kanji\'):<br />
```python main.py -d path/to/archive/kkanji/kkanji2 -l kanji```

- If you have downloaded and extracted the [Chinese MNIST dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist) and want to use the ResNet50 model for transfer learning, use the following command in your CLI (here with the label \'pretrained_resnet\'):<br />
```python main.py -d path/to/chinese-MNIST/data/data -c path/to/chinese-MNIST/chinese_mnist.csv -pm resnet50 -l pretrained_resnet -b False -rs 224 224 -g True```<br />
**HINT: Please be sure that you use -g True for pretrained models as ResNet50 and VGG16 because of the channels of the used images for training these models.**

- If you have run the main.py once and the saving of the data as npz files was successful, you can load the data from these npz files to save time (here with the label \'loaded_from_npz\'):<br />
```python main.py -ln True -np path/to/train_images.npz path/to/train_labels.npz path/to/test_images.npz path/to/test_labels.npz -l loaded_from_npz```

- If you have a local pretrained model that you want to use for transfer learning with  [Chinese MNIST dataset](https://www.kaggle.com/datasets/gpreda/chinese-mnist) you can use the following command in your CLI (with no balancing):<br />
```python main.py -d path/to/chinese-MNIST/data/data -c path/to/chinese-MNIST/chinese_mnist.csv -mp path/to/pretrained_model.h5 -l pretrained_model -b False```

## Model files

The used and trained models can be found [here](https://drive.google.com/drive/folders/1EqbCbd32bO3biHrdts9aO1wVDTHMXmib?usp=sharing).

## Github Link

The Github Link is: https://github.com/regnujAx/Deep-Learning-Kanji-Classification

## Dataset Links

Kuzushiji-49, Kuzushiji-MNIST, Kuzushiji-Kanji Datasets: https://www.kaggle.com/datasets/anokas/kuzushiji

Chinese-MNIST: https://www.kaggle.com/datasets/gpreda/chinese-mnist

## Notes Regarding demo file:

We made three demo files to represent our work with different datasets.

1) demo.ipynb  --Represents main demo file where we worked with Kuzushiji-Kanji and Chinese-MNIST datasets along with transfer learning

2) demo_kuzushiji_mnist_classification.ipynb  --Where we worked with Kuzushiji-MNIST dataset

3) demo_kuzushiji_49_classification.ipynb  --Where we worked with Kuzushiji-49 dataset