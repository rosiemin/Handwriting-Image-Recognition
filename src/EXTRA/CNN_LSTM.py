from src.read_data import IAMLoadData
from src.image_processing import IAM_imageprocess
import numpy as np
import pandas as pd
from keras.optimizers import Adadelta
from PIL import Image, ImageOps
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical
from keras.models import Model
from keras.layers.recurrent import LSTM
from src.parameter import *
import pdb

def get_model():
	inputs_shape = (img_h, img_w,1)     # (32, 64, 1)

	# model=Sequential()
	inputs = Input(name = "the_input", shape = inputs_shape)
	cnn_inner = Conv2D(64, (3, 3), padding = 'same', name = 'conv1', activation='relu')(inputs) # (None, 64, 32, 64)
	cnn_inner = (MaxPooling2D(pool_size=(2,2), name = 'pool1'))(cnn_inner) # (None,32, 16, 64)

	cnn_inner = Conv2D(128, (3, 3), padding = 'same', name = 'conv2', activation = 'relu')(cnn_inner) # (None, 32, 16, 128)
	cnn_inner = MaxPooling2D(pool_size=(2,2), name = 'pool2')(cnn_inner) # (None,16, 8, 128)

	cnn_inner = Conv2D(256, (3, 3), padding = 'same', name = 'conv3', activation = 'relu')(cnn_inner) # (None, 16, 8, 256)
	cnn_inner = Conv2D(256, (3, 3), padding = 'same', name = 'conv4', activation = 'relu')(cnn_inner) # (None, 16, 8, 256)

	cnn_inner = MaxPooling2D(pool_size=(1,2), name = 'pool3')(cnn_inner) # (None,16, 4, 256)
	cnn_inner = Conv2D(512, (3, 3), padding = 'same', name = 'conv5', activation = 'relu')(cnn_inner) # (None, 16, 4, 512)
	cnn_inner = Conv2D(512, (3, 3), padding = 'same', name = 'conv6', activation = 'relu')(cnn_inner) # (None, 16, 4, 512)

	lstm_inner = Reshape(target_shape=(16, 2048), name = 'reshape')(cnn_inner)
	lstm_inner = Dense(64, activation='relu', name = 'dense1')(lstm_inner)
	lstm_1 = LSTM(256, return_sequences=True, name = "lstm1")(lstm_inner)
	lstm_2 = LSTM(256, return_sequences=True, name = "lstm2")(lstm_1)

	inner = Dense((28), name = 'dense2')(lstm_2)
	y_pred = Activation('softmax', name = 'softmax')(inner)

	# Model(inputs = inputs, outputs = y_pred).summary()
	# labels = Input(name='the_labels', shape=[max_text_len], dtype='float32') # (None ,15)
	# input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
	# label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

	loss = ctc_utils.ctc_loss(output, labels, input_length)

	model = Model(inputs = [inputs], outputs = loss_out)
	y_func = K.function([inputs], [y_pred])

	return model, y_pred, y_func

def ctc_lambda_func(args):
	y_pred = args
	# the 2 is critical here since the first couple outputs of the RNN
	# tend to be garbage:
	y_pred = y_pred[:, 2:, :]
	return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def vectorize_word(sample):
    CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyz "

    characters = [letter for letter in CHAR_VECTOR]
    token_index = dict((char, characters.index(char)) for char in characters)
    max_length = 15
    results = np.zeros((len(sample), max_length, len(token_index.keys())+1))
    # pdb.set_trace()

    for i, samp in enumerate(sample):
        for k, char in enumerate(samp):
            index = token_index[char]
            results[i,k,index] = 1

    return results
if __name__ == '__main__':
	words = IAMLoadData('data/words.txt')
	df = words.load_data()
	clean = IAM_imageprocess(df, size = 100)
	X_train, X_test, y_train, y_test, df = clean.IAM_images()
	y_train = y_train[:,1]
	y_test = y_test[:,1]
	X_train = np.array(X_train)
	X_test = np.array(X_test)
	X_train_ex = np.expand_dims(X_train, axis = 3)
	X_test_ex = np.expand_dims(X_test, axis = 3)
	y_train_n = vectorize_word(y_train)
	y_test_n = vectorize_word(y_test)

	model, y_pred, y_func = get_model()
	ada = Adadelta()

	early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=10, mode='min', verbose=1)
	checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
	tensorboard = TensorBoard(log_dir='./logs', histogram_freq=2, batch_size=batch_size, write_graph=True, write_grads=True, write_images=True)

	model.compile(loss={'ctc': lambda y_train_n, y_pred: y_pred}, optimizer=ada)

	in_len = np.zeros(1,)
	label_len = np.zeros(1,)
	label = np.zeros(15)
	model.fit(X_train_ex,
	          verbose=1)

#
# CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyz "
#
# letters = [letter for letter in CHAR_VECTOR]
# letters = np.array(letters)
# num_classes = len(letters) + 1
#
# y_char = to_categorical(letters, num_classes)
# during fit process watch train and test error simultaneously
