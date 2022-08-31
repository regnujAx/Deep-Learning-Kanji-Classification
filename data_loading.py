import numpy as np
import os
import pandas as pd
import time

from keras.utils import to_categorical

from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


class DataLoader:
    def __init__(
            self, data_dir,
            balancing, upsampling_size,
            load_from_npz, npz_paths,
            save_to_npz, csv_file,
            use_grayscale, label,
            resize_shape):
        channels = 1
        if use_grayscale:
            channels = 3

        self.image_width, self.image_height = resize_shape
        self.input_shape = (self.image_width, self.image_height, channels)

        if load_from_npz:
            preprocessed_data = self.load_images_and_labels_from_npz(npz_paths, use_grayscale)
            self.num_classes = preprocessed_data[6]
        else:
            if csv_file != "":
                data = self.load_data_from_csv(data_dir, csv_file)
            else:
                data = self.load_data(data_dir)

            if balancing:
                data = self.balance_data(data, upsampling_size)

            images, labels = self.load_images_and_labels(data, resize_shape)
            self.num_classes = self.get_number_of_classes(labels)
            preprocessed_data = self.preprocess(images, labels, self.num_classes, save_to_npz, use_grayscale, label)

        self.train_images = preprocessed_data[0]
        self.train_labels = preprocessed_data[1]
        self.validation_images = preprocessed_data[2]
        self.validation_labels = preprocessed_data[3]
        self.test_images = preprocessed_data[4]
        self.test_labels = preprocessed_data[5]


    def load_data_from_csv(self, data_dir, csv_file):
        print("Load data from directory with csv file...")

        data_frame = pd.read_csv(csv_file)

        files = 0
        data = {}
        for _, row in data_frame.iterrows():
            label = row["code"]
            file_path = os.path.join(data_dir, f"input_{row['suite_id']}_{row['sample_id']}_{label}.jpg")

            files += 1

            if label in data:
                data[label].append(file_path)
            else:
                data[label] = [file_path]

        print(f"{files} files found")

        # Sort the data (needs Python 3.7 or higher)
        data = dict(sorted(data.items()))

        return data


    def load_data(self, data_dir):
        print("Load data from directory...")

        # Walk through the data directory and save all files in a dict
        files = 0
        data = {}
        for dir_path, _, file_names in os.walk(data_dir):
            label = os.path.basename(dir_path)
            for file in file_names:
                files += 1
                file_path = os.path.join(dir_path, file)
                if label in data:
                    data[label].append(file_path)
                else:
                    data[label] = [file_path]

        print(f"{files} files found")

        ### If you want to use only a subset of the collected data, 
        ### comment the following lines out and import random:
        ## print("Sample data...")
        ## files = 0
        ## sample_number = 1000
        ## sampled_data = dict(random.sample(data.items(), sample_number))

        ## # Save the selected labels
        ## f = open("labels.txt", "a")
        ## for label, images in sampled_data.items():
        ##     f.write(label + "\n")
        ##     files += len(images)

        ## print(f"{files} files selected")

        ## return sampled_data

        return data


    def load_images_and_labels_from_npz(self, npz_paths, use_grayscale):
        channels = 1
        if use_grayscale:
            channels = 3

        print("Load data from npz files...")

        train_images_path, train_labels_path, test_images_path, test_labes_path = npz_paths
        
        train_images = np.load(train_images_path)["arr_0"]
        train_labels = np.load(train_labels_path)["arr_0"]
        test_images = np.load(test_images_path)["arr_0"]
        test_labels = np.load(test_labes_path)["arr_0"]

        # Check if the train and test images have the correct shape
        if train_images.ndim != 4:
            train_images = np.expand_dims(train_images, axis=-1)
        if test_images.ndim != 4:
            test_images = np.expand_dims(test_images, axis=-1)

        print(f"\nNumber of training samples: {len(train_images)} where each sample is of size: {train_images.shape[1:]}")
        print(f"\nNumber of test samples: {len(test_images)} where each sample is of size: {test_images.shape[1:]}")

        # Check if the train and test labels are one-hot encoded
        if train_labels.ndim != 2:
            num_classes = len(np.unique(train_labels))
            train_labels = to_categorical(train_labels, num_classes)
        if test_labels.ndim != 2:
            num_classes = len(np.unique(test_labels))
            test_labels = to_categorical(test_labels, num_classes)
        num_classes = train_labels.shape[1]
        
        print("\nNumber of classes:", num_classes)

        # Split the train dataset in train and validation datasets
        train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size = 0.1)

        # Normalize the datasets
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], channels)/255
        validation_images = validation_images.reshape(validation_images.shape[0], validation_images.shape[1], validation_images.shape[2], channels)/255
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], channels)/255

        return train_images, train_labels, validation_images, validation_labels, test_images, test_labels, num_classes


    def balance_data(self, data, upsampling_size):
        print("Balance data...")

        # Up- and downsample imbalanced data to balance it
        for key, value in data.items():
            value_upsampled = resample(
                value,
                replace=True,
                n_samples=upsampling_size,
                random_state=123)
            data[key] = value_upsampled

        return data


    def load_images_and_labels(self, data, resize_shape):
        print("Load images and labels from data...")

        # Create 4 image arrays to speed up the data loading
        images1 = np.empty((0, self.image_width, self.image_height), int)
        images2 = np.empty((0, self.image_width, self.image_height), int)
        images3 = np.empty((0, self.image_width, self.image_height), int)
        images4 = np.empty((0, self.image_width, self.image_height), int)
        labels = np.array([], int)

        i = 0
        number_of_arrays = 4
        images_length = int(len(data.keys()) / number_of_arrays)
        print("Number of folders for one array:", images_length)

        start_time = time.time()
        for label in data.keys():
            images_chunk = np.empty((0, self.image_width, self.image_height), int)
            files = []

            print(f"Load from folder {i}...")
            for index, file in enumerate(data[label]):
                if file not in files:
                    img = Image.open(file)
                    img = np.array(img.resize(resize_shape))
                    files.append(file)
                else:
                    img = images_chunk[index-1]
                images_chunk = np.append(images_chunk, img.reshape(1, self.image_width, self.image_height), axis=0)
                labels = np.append(labels, i)

            if i < images_length:
                images1 = np.concatenate((images1, images_chunk))
            elif i < images_length * 2:
                images2 = np.concatenate((images2, images_chunk))
            elif i < images_length * 3:
                images3 = np.concatenate((images3, images_chunk))
            else:
                images4 = np.concatenate((images4, images_chunk))

            i = i + 1

        images = np.concatenate((images1, images2, images3, images4))

        print(f"\nNumber of samples: {len(images)} where each sample is of size: {images.shape[1:]}")
        elapsed_time = time.time() - start_time
        time_string = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print("\nElapsed time for data preparation:", time_string)

        # One-hot encode labels
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(labels)

        return images, labels


    def get_number_of_classes(self, labels):
        num_classes = len(np.unique(labels))
        print("\nNumber of classes:", num_classes)

        return num_classes


    def preprocess(self, images, labels, num_classes, save_to_npz, use_grayscale, label):
        channels = 1
        if use_grayscale:
            # Add additional channels to grayscale images (for pretrained models)
            channels = 3
            images = np.repeat(images[..., np.newaxis], channels, -1)
        else: 
            # Use expand_dims to get a nominal deep learning format for all images
            # (64, 64) --> (64, 64, 1)
            images = np.expand_dims(images, axis=-1)

        # Convert label vector to matrix
        labels = to_categorical(labels, num_classes)

        # Split the whole dataset in train and test datasets (images and labels)
        train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.1)
        print("Data splitted in train and test sets")

        if save_to_npz:
            print("Save train and test images and labels to npz files...")
            np.savez(f"train_images_{label}.npz", train_images)
            np.savez(f"test_images_{label}.npz", test_images)
            np.savez(f"train_labels_{label}.npz", train_labels)
            np.savez(f"test_labels_{label}.npz", test_labels)

        # Split the train dataset in train and validation datasets (images and labels)
        train_images, validation_images, train_labels, validation_labels = train_test_split(train_images, train_labels, test_size=0.1)
        print("Train set splitted in train and validation sets")

        # Normalize the datasets
        train_images = train_images.reshape(train_images.shape[0], train_images.shape[1], train_images.shape[2], channels)/255
        validation_images = validation_images.reshape(validation_images.shape[0], validation_images.shape[1], validation_images.shape[2], channels)/255
        test_images = test_images.reshape(test_images.shape[0], test_images.shape[1], test_images.shape[2], channels)/255
        
        # Print the dimensions of the datasets
        print(f"\nTrain images dimensions: {train_images.shape}")
        print(f"\nValidation images dimensions: {validation_images.shape}")
        print(f"\nTest images dimensions: {test_images.shape}")

        return train_images, train_labels, validation_images, validation_labels, test_images, test_labels
