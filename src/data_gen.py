import os
import numpy as np
import cv2
import io
import json
import keras
import string

from src.preprocess import read_image, norm_img


class DataGenerator(keras.utils.Sequence):

    def __init__(self, config, dataset, shuffle=True, use_data_augmentation=False):
        """
        Constructor
        """

        self.config = config
        self.dataset = dataset
        self.dataset_len = len(dataset)
        self.images_folder = self.config['images_folder']
        self.batch_size = self.config['train']['batch_size']
        self.y_size = self.config['image']['image_size']['y_size']
        self.x_size = self.config['image']['image_size']['x_size']
        self.num_channels = self.config['image']['image_size']['num_channels']
        self.shuffle = shuffle
        self.decoder_tokens = sorted(string.printable)
        self.num_decoder_tokens = len(self.decoder_tokens)
        self.max_seq_length = self.config['network']['max_seq_lenght']
        self.token_indices, self.reverse_token_indices = self.token_indices()
        self.indices = np.arange(self.dataset_len)

    def __len__(self):

        return int(np.floor(self.dataset_len / self.batch_size))

    def __getitem__(self, index):

        # Generate indexes of the batch
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        dataset_temp = [self.dataset[k] for k in indices]

        # Generate data
        X, y1, y2 = self.data_generation(dataset_temp)

        return [X, y1], y2

    def on_epoch_end(self):

        if self.shuffle == True:
            np.random.shuffle(self.indices)

    def get_full_dataset(self):

        dataset_images = []
        dataset_labels = []

        for elem in self.dataset:
            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']
            num_channels = self.config['image']['image_size']['num_channels']

            #read image
            image = read_image(self.images_folder, elem['filename'], y_size, x_size)
            image = norm_img(image, y_size, x_size)

            label = elem['label']

            dataset_images.append(image)
            dataset_labels.append(label)

        dataset_images = np.asarray(dataset_images)

        return dataset_images, dataset_labels


    def data_generation(self, dataset_temp):

        batch_x = []
        batch_y1 = []
        batch_y2 = []

        for elem in dataset_temp:
            y_size = self.config['image']['image_size']['y_size']
            x_size = self.config['image']['image_size']['x_size']
            num_channels = self.config['image']['image_size']['num_channels']


            image = read_image(self.images_folder, elem['filename'], y_size, x_size)
            image = norm_img(image, y_size, x_size)

            decoder_input_data, decoder_target_data = self.one_hot_labels(elem['label'], self.max_seq_length,
            self.num_decoder_tokens,
            self.token_indices)

            batch_x.append(image)
            batch_y1.append(decoder_input_data)
            batch_y2.append(decoder_target_data)

        batch_x = np.asarray(batch_x, dtype = np.float32)
        batch_y1 = np.asarray(batch_y1, dtype = np.float32)
        batch_y2 = np.asarray(batch_y2, dtype = np.float32)

        return batch_x, batch_y1, batch_y2

    def token_indices(self):

        target_token_index = dict((k, v) for v, k in enumerate(self.decoder_tokens))
        reverse_target_token_index = dict((i, char) for char, i in target_token_index.items())

        return target_token_index, reverse_target_token_index

    def one_hot_labels(self, label, max_seq_length, num_decoder_tokens, target_token_index):

        decoder_input_data = np.zeros((max_seq_length, num_decoder_tokens), dtype='float32')
        decoder_target_data = np.zeros((max_seq_length, num_decoder_tokens), dtype='float32')

        #generate one hot label for input decoder
        for t, char in enumerate('[' + label):
            if t < max_seq_length:
                decoder_input_data[t, target_token_index[char]] = 1.

        #generate one hot label for output decoder
        for t, char in enumerate(label + ']'):
            if t < max_seq_length:
                decoder_target_data[t, target_token_index[char]] = 1.

        return decoder_input_data, decoder_target_data
