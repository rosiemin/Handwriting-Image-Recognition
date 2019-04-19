import os
import numpy as np
import cv2
import random
import datetime
import io
import json
import keras
import string


from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense, TimeDistributed, Conv2D, MaxPooling2D, Reshape, Dropout, BatchNormalization, Activation, Bidirectional, concatenate, add, Lambda, Permute
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau


class Trainer(object):

    def __init__(self, config, model, train_generator, val_generator):
        self.config = config
        self.model = model.model
        self.train_generator = train_generator
        self.val_generator = val_generator
        self.epochs = self.config['train']['num_epochs']
        self.callbacks_list = self.callbacks()


    def callbacks(self):

        callbacks = []

        #early stopping
        if self.config['callbacks']['early_stopping']['enabled'] == True:
            monitor = self.config['callbacks']['early_stopping']['monitor']
            patience = self.config['callbacks']['early_stopping']['patience']
            callbacks.append(EarlyStopping(monitor=monitor, min_delta=0, patience=patience, verbose=1, mode='auto'))

        #tensorboard
        if self.config['callbacks']['tensorboard']['enabled'] == True:
            log_dir = self.config['callbacks']['tensorboard']['log_dir']
            callbacks.append(TensorBoard(log_dir=log_dir))

        #best checkpoint
        if self.config['callbacks']['model_best_checkpoint']['enabled'] == True:
            monitor = self.config['callbacks']['model_best_checkpoint']['monitor']
            filepath = self.config['callbacks']['model_best_checkpoint']['out_file']
            callbacks.append(ModelCheckpoint(filepath, monitor=monitor, verbose=1, save_best_only=True,
                                             save_weights_only=True, mode='min'))

        #last checkpoint
        if self.config['callbacks']['model_last_checkpoint']['enabled'] == True:
            filepath = self.config['callbacks']['model_last_checkpoint']['out_file']
            callbacks.append(ModelCheckpoint(filepath, verbose=1, save_best_only=False,
                            save_weights_only=True))

        return callbacks


    def train(self):

        use_multiprocessing = self.config['train']['use_multiprocessing']
        num_workers = self.config['train']['num_workers']

        H = self.model.fit_generator(generator=self.train_generator, validation_data=self.val_generator,
                                 epochs=self.epochs, verbose=1, max_queue_size=10, workers=num_workers,
                                 use_multiprocessing=use_multiprocessing, shuffle=False,
                                 callbacks=self.callbacks_list)

        graph_path = self.config['train']['output']['output_graph']
        weights_path = self.config['train']['output']['output_weights']

        print("Saving graph and weights in", graph_path, ",", weights_path)
        self.save_model(self.model, graph_path, weights_path)

        return H

    def save_model(self, model, graph_path, weights_path):

        model_json = model.to_json()
        with open(graph_path, "w") as json_file:
            json_file.write(model_json)

        model.save_weights(weights_path)
