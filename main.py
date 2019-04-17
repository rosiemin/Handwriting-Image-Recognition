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
from src.preproc_functions import read_image_BW, normalize_0_mean_1_variance_BW


def train(args):
    """
    Train a model on the train set defined in labels.json
    """

    config_path = args.conf

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

    config_path = args.conf

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


def predict(args, filename):
    """
    Predict on a single image
    """

    config_path = args.conf

    filename = args.filename

    with open(config_path) as f:
        config = yaml.load(f)

    with open(config['labels_file']) as f:
        dataset = json.load(f)

    test_generator = DataGenerator(config, dataset['test'], shuffle=False, use_data_augmentation=False)

    y_size = config['image']['image_size']['y_size']
    x_size = config['image']['image_size']['x_size']
    num_channels = config['image']['image_size']['num_channels']
    convert_to_grayscale = config['image']['convert_to_grayscale']

    #read image
    if num_channels == 1 or (num_channels == 3 and convert_to_grayscale):
        image = read_image_BW('./', filename, y_size, x_size)
        image = normalize_0_mean_1_variance_BW(image, y_size, x_size)
        image = np.reshape(image, (1, y_size, x_size, 1))
    else:
        image = read_image_color('./', filename, y_size, x_size)
        image = normalize_0_mean_1_variance_color(image, y_size, x_size)
        image = np.reshape(image, (1, y_size, x_size, 3))

    #print(image.shape)

    graph_file =  config['predict']['graph_file']
    weights_file = config['predict']['weights_file']
    batch_size = 1

    predictor = PredictorSeq2Seq(config, graph_file, weights_file, test_generator.num_decoder_tokens, test_generator.max_seq_length, test_generator.token_indices, test_generator.reverse_token_indices,
                                 batch_size = batch_size)

    pred = predictor.predict(image)
    print(pred)
    print("Predicted label:", pred[0])

if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='Seq2seq')
    # parser.add_argument('-c', '--conf', help='path to configuration file', required=True)


    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', help='Train')
    group.add_argument('--predict_on_test', action='store_true', help='Predict on test set')
    group.add_argument('--predict', action='store_true', help='Predict on single file')

    parser.add_argument('--filename', help='path to file')

    args = parser.parse_args()

    #    print(args)

    if args.predict_on_test:
        print('Predicting on test set')
        predict_on_test(args)

    elif args.predict:
        if args.filename is None:
            raise Exception('missing --filename FILENAME')
        else:
            print('predict')
            predict(args)

    elif args.train:
        print('Starting training')
        train(args)
    else:
        raise Exception('Unknown args')
