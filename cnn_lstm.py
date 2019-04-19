import argparse
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

    config_path = args.conf

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

def score_prediction(y_true, y_pred):

    words_identified = 0
    characters_identified = 0
    char_tot = 0

    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i]:
            words_identified += 1

        levenshtein_distance = edit_distance(y_true[i], y_pred[i])
        n_char = np.maximum(len(y_true[i]), len(y_pred[i]))

        normalized_distance = levenshtein_distance/n_char

        characters_identified += normalized_distance

    CER = float((characters_identified) / len(y_true)) #character error rate
    WER = (len(y_pred) - words_identified)/len(y_pred) #word error rate

    return CER, WER

def predict_on_test(args):
    """
    Predict on the test set defined in labels.json
    """

    config_path = args.conf

    with open(config_path) as f:
        config = yaml.load(f)

    with open(config['labels_file']) as f:
        dataset = json.load(f)

    test_generator = DataGenerator(config, dataset['test'], shuffle=False)

    images_test, labels_test = test_generator.get_full_dataset()

    graph_file =  config['predict']['graph_file']
    weights_file = config['predict']['weights_file']
    batch_size = config['predict']['batch_size']

    predictor = Predictor(config, graph_file, weights_file, test_generator.num_decoder_tokens, test_generator.max_seq_length, test_generator.token_indices, test_generator.reverse_token_indices, batch_size = batch_size)

    pred_test = predictor.predict(images_test)

    CER, WER  = score_prediction(labels_test, pred_test)

    for i in range(len(labels_test)):
        print(labels_test[i], pred_test[i])

    print('CER: ', round(CER * 100, 2), '%')
    print('WER: ', round(WER * 100, 2), '%')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CNN-LSTM')
    parser.add_argument('-c', '--conf', help='path to configuration file', required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')
    group.add_argument('--predict', action='store_true', help='Predict on test set')
    args = parser.parse_args()


    if args.predict:
        print('Predicting on test set')
        predict_on_test(args)
    elif args.train:
        print('Starting training')
        train(args)
