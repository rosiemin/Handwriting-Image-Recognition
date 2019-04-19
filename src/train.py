import os
import numpy as np
import json
import yaml
from nltk.metrics.distance import edit_distance
from src.data_gen import DataGenerator
from src.model import Models
from src.trainer import Trainer
from src.predictor import Predictor
from src.preprocess import read_image, norm_img


def train(args):
    """
    Train a model on the train set defined in labels.json
    """

    config_path = args

    with open(config_path) as f:
        config = yaml.load(f)

    with open(config['labels_file']) as f:
        dataset = json.load(f)

    train_generator = DataGenerator(config, dataset['train'], shuffle=True)

    val_generator = DataGenerator(config, dataset['val'], shuffle=True)

    max_seq_length = train_generator.max_seq_length
    num_decoder_tokens = train_generator.num_decoder_tokens

    train_model = Models(config, max_seq_length, num_decoder_tokens)
    trainer = Trainer(config, train_model, train_generator, val_generator)

    H = trainer.train()
    return H

if __name__ == '__main__':

    print('Starting training')
    H = train('src/config.yml')
