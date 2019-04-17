import argparse
import os
import numpy as np
import json
import yaml
from nltk.metrics.distance import edit_distance
import numpy as np
from src.data_generator_seq2seq import DataGenerator
from src.model_seq2seq import ModelSeq2Seq
from src.trainer_seq2seq import TrainerSeq2Seq
from src.predictor_seq2seq import PredictorSeq2Seq
# from util import score_prediction
from src.preproc_functions import read_image_BW, normalize_0_mean_1_variance_BW
import matplotlib.pyplot as plt


def train(args):
    """
    Train a model on the train set defined in labels.json
    """

    config_path = args

    with open(config_path) as f:
        config = yaml.load(f)

    with open(config['labels_file']) as f:
        dataset = json.load(f)

    #----------train generator--------
    if config['train']['train_on_subset']['enabled']:
        #select only a fraction
        fraction = config['train']['train_on_subset']['dataset_fraction']
        max_len = int(len(dataset['train']) * fraction)

        np.random.seed(config['train']['train_on_subset']['random_seed'])
        indices = np.random.randint(0, len(dataset['train']), max_len)
        dataset['train_subsampled'] = [dataset['train'][j] for j in indices]

        train_generator = DataGenerator(config, dataset['train_subsampled'], shuffle=True,
                                    use_data_augmentation=config['data_aug']['use_data_aug'])
    else:
        train_generator = DataGenerator(config, dataset['train'], shuffle=True,
                                    use_data_augmentation=config['data_aug']['use_data_aug'])

    #----------val generator--------
    val_generator = DataGenerator(config, dataset['val'], shuffle=True, use_data_augmentation=False)

    max_seq_length = train_generator.max_seq_length
    num_decoder_tokens = train_generator.num_decoder_tokens

    train_model = ModelSeq2Seq(config, max_seq_length, num_decoder_tokens)
    trainer = TrainerSeq2Seq(config, train_model, train_generator, val_generator)

    H = trainer.train()
    return H

if __name__ == '__main__':

    print('Starting training')
    H = train('src/config.yml')

    plt.plot(range(len(H.history['loss'])), H.history['loss'], label = 'Training Loss')
    plt.plot(range(len(H.history['val_loss'])), H.history['val_loss'], label = 'Validation Loss')
    plt.xlabel('Number of Epochs')
    plt.ylabel('Loss (Categorical Crossentropy)')
    plt.title('Loss over Epochs for CNN-LSTM')
    plt.legend()
    plt.savefig('loss.png')
