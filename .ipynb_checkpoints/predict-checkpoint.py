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


def score_prediction(y_true, y_pred):
    """Function to score prediction on IAM, using Levenshtein distance
       to calculate character error rate (CER)

    Parameters
    ------
    y_true: list
        list of ground truth labels
    y_pred: list
        list of predicted labels

    Returns
    -------
    CER: float
        character error rate
    WER: float
        word error rate
    """

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

    # array_accuracy_characters = np.asarray(list_accuracy_characters)
    CER = float((characters_identified) / len(y_true))
    WER = (len(y_pred) - words_identified)/len(y_pred)

    return CER, WER

def predict_on_test(args):
    """
    Predict on the test set defined in labels.json
    """

    config_path = args

    with open(config_path) as f:
        config = yaml.load(f)

    with open(config['labels_file']) as f:
        dataset = json.load(f)

    test_generator = DataGenerator(config, dataset['test'], shuffle=False, use_data_augmentation=False)

    #numpy array containing images
    images_test, labels_test = test_generator.get_full_dataset()

    #print(images_test.shape)
    #print(len(labels_test))

    graph_file =  config['predict']['graph_file']
    weights_file = config['predict']['weights_file']
    batch_size = config['predict']['batch_size']

    predictor = PredictorSeq2Seq(config, graph_file, weights_file, test_generator.num_decoder_tokens, test_generator.max_seq_length, test_generator.token_indices, test_generator.reverse_token_indices,
                                 batch_size = batch_size)

    pred_test = predictor.predict(images_test)

    CER, WER  = score_prediction(labels_test, pred_test)

    for i in range(len(labels_test)):
        print(labels_test[i], pred_test[i])

    print('CER: ', round(CER * 100, 2), '%')
    print('WER: ', round(WER * 100, 2), '%')

    return CER, WER, labels_test, pred_test, images_test, datatset['test']

if __name__ == '__main__':
    print('Predicting on test set')
    CER, WER, labels_test, pred_test, images_test, dataset['test'] = predict_on_test('src/config.yml')
