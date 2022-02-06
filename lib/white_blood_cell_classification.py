import csv
import logging
import os

import pandas as pd
import numpy as np
import cv2
import skimage.morphology as morph
import multiprocessing as mp
import tensorflow as tf

from keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class WhiteBloodCellClassification:
    """
    This class is a CNN abstraction in order to classify white blood cell. It use a simple CNN and
    the main work has been done on image pre processing.
    """
    def __init__(self, use_gpu=True, force=False,
                 raw_train='data/raw/train',
                 raw_test='data/raw/test',
                 train_folder='data/train',
                 validation_folder='data/validation',
                 test_folder='data/test'):
        """
        This function will
        :param use_gpu: If the use of GPU is not possible set this boolean to False, it will tell
        keras to only use the processor
        :param force: If set to True the data preprocessing will overwrite pre existing one
        :param raw_train: Use to change the path of raw training data
        :param raw_test: Use to change the path of raw test data
        :param train_folder: Use to change the target folder for pre processed train data
        :param validation_folder: Use to change the target folder for pre processed validation data
        :param test_folder: Use to change the target folder for pre processed test data
        """
        self.model = None
        if not use_gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        self.force = force
        self.types = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        # source folder from the dataset
        self.raw_train = raw_train
        self.raw_test = raw_test

        # our folder for each set
        self.train_folder = train_folder
        self.validation_folder = validation_folder
        self.test_folder = test_folder

        self.train_generator = None
        self.test_generator = None

    def _create_dataset(self):
        """
        This function will create the training, validation and testing dataset from the original
        ones.
        The dataset used to train this part of the program is divided in two part train and test
        which follow the same architecture. In each folder there is 5 sub-folder corresponding to
        each class we want to predict: Basophil, Eosinophiln Lymphocyte, Monocyte and Neutrophil.
        Each image has a resolution of 500x500 which is too big for us, we want to resize them to
        128x128 (this will also permit to optimize GPU memory). In order to do so we will walk into
        each directory and load each image one at a time. We crop them and then we use morphology
        mathematics to extract only the kernel of the cell which is what we used to differentiate
        each type of cell. You can learn more about this process in the main report. Once this is
        done we save the picture into a new directory and then write a file called "class.csv" which
        contain each picture name associated with the class. This will be used after to create
        Datagenerator.
        :return:
        """
        logging.info("Starting dataset creation...")

        if not os.path.isdir('data/train') or self.force is True:

            logging.info("No train folder detected...")
            logging.info("Creating folders...")

            if not os.path.isdir(self.train_folder):
                os.mkdir(self.train_folder)
                os.mkdir(self.test_folder)

            # Load Train
            train_csv = open(self.train_folder + '/class.csv', 'w', newline='')
            test_csv = open(self.test_folder + '/class.csv', 'w', newline='')
            train_writer = csv.writer(train_csv)
            test_writer = csv.writer(test_csv)
            train_writer.writerow(['Image', 'Id'])
            test_writer.writerow(['Image', 'Id'])
            i = 1
            for c in self.types:
                logging.info(c)
                folder_size = len([name for name in os.listdir(self.raw_train + c) if os.path.isfile(name)])
                count = 0
                for f in os.listdir(self.raw_train + c):
                    # if os.path.isfile('data/' + c):
                    p = self.get_core(self.resize(self.raw_train + c + '/' + f, open_file=True))
                    cv2.imwrite(self.train_folder + '/' + f, p)
                    # shutil.copy(raw_train + c + '/' + f, train_folder + '/' + f)
                    train_writer.writerow([f, i])
                    count += 1
                    # if train_size >= folder_size * train_size:
                    #     break
                count = 0
                for f in os.listdir(self.raw_test + c):
                    # if os.path.isfile('data/' + c):
                    p = self.get_core(self.resize(self.raw_test + c + '/' + f, open_file=True))
                    cv2.imwrite(self.test_folder + '/' + f, p)
                    # shutil.copy(raw_train + c + '/' + f, train_folder + '/' + f)
                    test_writer.writerow([f, i])
                    # shutil.copy(raw_test + c + '/' + f, test_folder + '/' + f)
                    # test_writer.writerow([f, i])
                i += 1
            train_csv.close()
            test_csv.close()
        logging.info("Dataset creation successful.")

    def _init_generator(self):
        logging.info("Creating data generator...")
        self.train_generator = self.create_generator(self.train_folder, 'training')
        # validation_generator = create_generator(validation_folder, 'validation')
        self.test_generator = self.create_generator(self.test_folder, 'validation')
        logging.info("Creation successful.")

    def _create_model(self):
        self.model = models.Sequential()
        self.model.add(
            layers.Conv2D(128, activation='relu', input_shape=(128, 128, 3), kernel_size=(5, 5),
                          strides=1))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(256, (5, 5), activation='relu'))
        self.model.add(layers.MaxPooling2D((2, 2)))
        self.model.add(layers.Conv2D(512, (5, 5), activation='relu'))
        self.model.add(layers.Dropout(0.25))
        self.model.add(Flatten())
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(5, activation='softmax'))
        self.model.compile(optimizer='adam', metrics=['accuracy'],
                           loss=tf.keras.losses.categorical_crossentropy)
        self.model.save('model')

    def train(self):
        """
        This function will train the CNN, it will create the dataset if it was not previously
        created.
        :return:
        """
        logging.info("Starting training...")
        self._create_dataset()
        self._init_generator()
        self._create_model()
        logging.info("Starting fit...")
        self.model.fit(self.train_generator, validation_data=self.train_generator, epochs=10,
                       verbose=2)
        self.model.save('model')
        logging.info("Training successful!")

    def predict(self, data, is_open=True, model_path='model', train_upon_error=False):
        """
        This function take as an input an 128x128 image of a white blood cell and will return it
        class. The class is a number between 0 and 5 which can used is the function "get_type" to
        get the white blood cell type.
        :param train_upon_error: If set to True the function will train the model if the model
        has not been found
        :param data: A 128x128 image
        :param is_open: If the image is already open set the boolean to False
        :param model_path: The path of the pre-train model. Leave it empty if the model hasn't been
        train or hasn't been move
        :return: A number between 0 and 5 corresponding to the class, None upon error
        """
        if not is_open:
            data = cv2.imread(data)
        if not self.model or model_path is not 'model':
            logging.info("Model has not been initialized...")
            if os.path.isdir(model_path):
                logging.info("Model found, it will be loaded.")
                self.model = keras.models.load_model(model_path)
            else:
                logging.warning("Model not found while using predict, is the repository correct ?")
                if train_upon_error:
                    logging.warning("train_upon_error is True, model will be train...")
                    self.train()
                else:
                    logging.warning("train_upon_error is False, model will not be train...")
                    return None
        logging.info("Predicting class...")
        print("-----------------------------")
        print(data.shape)
        predicted_class = self.model.predict(data)
        logging.info("Predicted: " + str(predicted_class))
        return predicted_class

    def get_type(self, predicted_class):
        """
        This function should be used to interprete the result of predict function. Upon error
        it will return None and a string containing the type otherwise.
        :param predicted_class: Result of predict function.
        :return: A string upon success, None otherwise
        """
        if predicted_class < 0 or predicted_class > 5:
            logging.error("Class is not between bound! Must be between 0 and 5")
            return None
        return self.types[predicted_class]

    @staticmethod
    def create_generator(folder, subset):
        """
        This function can be used to convert a folder containing a class.csv and create a generator
        to use with the model
        :param folder: the folder you want to be convert
        :param subset: train, validation or None for test
        :return: The corresponding dataset
        """
        df = pd.read_csv(folder + '/class.csv', dtype=str, delimiter=',')
        data = ImageDataGenerator(rescale=1. / 255.)
        return data.flow_from_dataframe(
            dataframe=df,
            directory=folder,
            x_col='Image',
            y_col='Id',
            subset=subset,
            batch_size=128,
            shuffle=True,
            class_mode='categorical',
            target_size=(128, 128),
            validation_split=0.25
        )

    @staticmethod
    def get_grayscale(cell, ori=None):
        ori = np.copy(cell) 
        cell = np.dot(cell[..., :3], [0.2999, 0.587, 0.114]) 
        cell = cell > 100
        return cell, ori

    @staticmethod
    def get_core(cell, ori=None):
        """
        Use to extract only the kernel of the cell
        :param cell:
        :param ori:
        :return:
        """
        if type(cell) == tuple :
            cell, ori = cell[0], cell[1]
        elif ori is None:
            cell, ori = cell, np.copy(cell)
        shape = np.uint8(np.invert(morph.remove_small_holes(cell, 1024)))
        #plt.imshow(shape, cmap='gray')
        zeros = np.zeros((128, 128))
        shape = np.stack((zeros + shape, shape, zeros + shape), axis=2)
        return np.uint8(ori * shape)

    @staticmethod
    def resize(cell, open_file=False, crop=True, resize=True):
        """
        Primarily used for training since the source shape of the image are too big (500x500) we convert
        it to a 128x128 shape
        :param cell: the source file
        :param open_file:  is the file currently open
        :param crop: default True, if the resolution is already 128x128 it should be se to False
        :param resize: default True, resize the image to 128x128
        :return: the image in grayscale and the origin
        """
        if open_file:
            cell = cv2.imread(cell)
        shape = cell.shape
        if crop:
            # Cropping the picture
            cell = np.delete(cell, np.s_[shape[0] - 150:shape[0]], axis=0)
            cell = np.delete(cell, np.s_[0:150], axis=0)
            cell = np.delete(cell, np.s_[shape[0] - 150:shape[0]], axis=1)
            cell = np.delete(cell, np.s_[0:150], axis=1)
        if resize:
            # resizing
            cell = cv2.resize(cell, (128, 128), interpolation=cv2.INTER_AREA)
        ori = np.copy(cell)
        cell = np.dot(cell[..., :3], [0.2999, 0.587, 0.114])
        cell = cell > 100
        return cell, ori

