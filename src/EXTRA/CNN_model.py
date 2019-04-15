from src.read_data import IAMLoadData
from src.image_processing import IAM_imageprocess
import numpy as np
import pandas as pd
from keras.optimizers import Adadelta, Adam
from keras.losses import categorical_crossentropy, binary_crossentropy
from PIL import Image, ImageOps
from keras import backend as K
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Dense, Activation, Flatten, Dropout
from keras.layers import Reshape, Lambda, BatchNormalization
from keras.layers.merge import add, concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.utils import to_categorical, plot_model
from keras.models import Model
import pdb
import string
from keras.preprocessing.text import one_hot
from keras.callbacks import History
import matplotlib.pyplot as plt
from mlxtend.evaluate import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix




def onehotencode_targets(lst_of_words):
    CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyz "

    letters = [letter for letter in CHAR_VECTOR]

    alphabet_df = pd.DataFrame(columns = letters, index = lst_of_words)
    alphabet_df.fillna(0, inplace = True)
    # pdb.set_trace()
    for word in lst_of_words:
        for letter in word:
            for col in alphabet_df:
                if letter == col:
                    alphabet_df.loc[word, col] = 1

    return alphabet_df
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


def get_model():
    n_classes = 27
    inputs_shape = (32, 64 ,1)
    # model=Sequential()
    inputs = Input(name = "the_input", shape = inputs_shape)
    cnn_inner = Conv2D(64, (3, 3), padding = 'same', name = 'conv1', activation='relu')(inputs) # (None, 64, 32, 64)
    cnn_inner = Conv2D(64, (3, 3), padding = 'same', name = 'conv1.5', activation='relu')(cnn_inner) # (None, 64, 32, 64)
    cnn_inner = (MaxPooling2D(pool_size=(2,2), name = 'pool1'))(cnn_inner) # (None,32, 16, 64)
    cnn_inner = Dropout(0.20)(cnn_inner)

    cnn_inner = Conv2D(128, (3, 3), padding = 'same', name = 'conv2', activation = 'relu')(cnn_inner) # (None, 32, 16, 128)
    cnn_inner = MaxPooling2D(pool_size=(2,2), name = 'pool2')(cnn_inner) # (None,16, 8, 128)
    cnn_inner = Dropout(0.50)(cnn_inner)

    cnn_inner = Conv2D(256, (3, 3), padding = 'same', name = 'conv3', activation = 'relu')(cnn_inner) # (None, 16, 8, 256)
    cnn_inner = Conv2D(256, (3, 3), padding = 'same', name = 'conv4', activation = 'relu')(cnn_inner) # (None, 16, 8, 256)
    cnn_inner = MaxPooling2D(pool_size=(1,2), name = 'pool3')(cnn_inner) # (None,16, 4, 256)
    cnn_inner = Dropout(0.20)(cnn_inner)

    cnn_inner = Conv2D(512, (3, 3), padding = 'same', name = 'conv5', activation = 'relu')(cnn_inner) # (None, 16, 4, 512)
    cnn_inner = Conv2D(512, (3, 3), padding = 'same', name = 'conv6', activation = 'relu')(cnn_inner) # (None, 16, 4, 512)

    cnn_flat = Flatten(name = 'flatten')(cnn_inner)
    dense_hidden = Dense(2048, activation='relu', name = 'dense')(cnn_flat)
    dense_hidden = Dropout(0.20)(dense_hidden)
    y_pred = Dense(n_classes, activation='sigmoid', name = 'char_pred')(dense_hidden)
    # num_pred = Dense(1, activation = 'relu', name = 'num_char')(dense_hidden)

    # # sequence model
    # inputs2 = Input(shape=(20,))
    # se1 = Embedding(n_classes, 256, mask_zero=True)(inputs2)
    # lstm_1 = LSTM(256, return_sequences=True, name = "lstm1")(se1)# decoder model
    # decoder1 = add([dense_hidden, lstm_1])
    # decoder2 = Dense(256, activation='relu')(decoder1)
    # outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # # tie it together [image, seq] [word]
    # model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    # #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    # sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
    # model.compile(loss='categorical_crossentropy', optimizer='sgd') #metrics=['accuracy']
    # # summarize model
    # print(model.summary())

    model = Model(inputs = inputs, outputs = y_pred)

    return model, y_pred

def precision(y_true, y_pred, threshold_shift=0.5):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    return precision


def recall(y_true, y_pred, threshold_shift=0.5):
    beta = 1

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred_bin, 0, 1)))

    recall = tp / (tp + fn)
    return recall


def fbeta(y_true, y_pred, threshold_shift=0.5):
    beta = 2

    # just in case of hipster activation at the final layer
    y_pred = K.clip(y_pred, 0, 1)

    # shifting the prediction threshold from .5 if needed
    y_pred_bin = K.round(y_pred + threshold_shift)

    tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
    fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    beta_squared = beta ** 2
    return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)


if __name__ == '__main__':
    words = IAMLoadData('data/words.txt')
    df = words.load_data()
    clean = IAM_imageprocess(df, size = 2000)
    X_train, X_test, y_train, y_test, df = clean.IAM_images()
    y_train_code = onehotencode_targets(y_train[:,1].tolist()).values
    y_test_code = onehotencode_targets(y_test[:,1].tolist()).values
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    X_train_lst = np.expand_dims(X_train, axis = 3)
    X_test_lst = np.expand_dims(X_test, axis = 3)
    y_train_v = vectorize_word(y_train[:,1])
    y_test_v = vectorize_word(y_test[:,1])

    early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=6, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath='model/CNN--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
    tensorboard = TensorBoard(log_dir='./logs', batch_size=100, write_graph=True, write_grads=True, write_images=True, histogram_freq=1)

    model, y_pred = get_model()
    ada = Adadelta()
    adam = Adam(lr = 0.00001)
    model.compile(loss = 'binary_crossentropy',
              optimizer=adam,
              metrics=['accuracy', precision, fbeta, recall])

    history = model.fit(X_train_lst, y_train_code, batch_size=100, epochs=50,  verbose=1, callbacks=[early_stop, tensorboard], validation_data=(X_test_lst, y_test_code))#, validation_data=(X_test_lst, y_test_code))
    score = model.evaluate(X_test_lst, y_test_code, verbose = 0)

    acc = history.history['acc']
    loss = history.history['loss']
    epochs = range(1, len(acc) + 1)


    idx = ['y_true',' 0.15', '0.16', '0.17', '0.18', '0.19', '0.2 ', '0.21', '0.22', '0.23', '0.24', '0.25',
       '0.26', '0.27', '0.28', '0.29', '0.3 ',' 0.31', '0.32', '0.33', '0.34', '0.35', '0.36',
       '0.37', '0.38',' 0.39', '0.4' , '0.41', '0.42', '0.43', '0.44', '0.45', '0.46', '0.47']

    thres = np.linspace(0.15, 0.47, 33)
    df_pred = pd.DataFrame(columns = letters)
    df_pred.loc[len(df_pred)] = np.sum(y_test_code, axis = 0)
    #
    for j,i in zip(idx,thres):
        df_pred.loc[len(df_pred)] = np.sum(np.where(preds_y_test>= i, 1, 0), axis = 0)
    df_pred.index = idx

    plt.plot(H.history['acc'],color = 'turquoise')
    plt.plot(H.history['val_acc'], color= 'navy', linestyle = '-.')
    plt.title('Model accuracy', fontsize=18)
    plt.ylabel('Accuracy',fontsize=16)
    plt.xlabel('Epoch',fontsize=16)
    plt.legend(['Train', 'Test'], loc='upper left', fontsize=16)
    plt.show()

    # Plot training & validation loss values
    plt.plot(H.history['loss'], color= 'turquoise')
    plt.plot(H.history['val_loss'], color= 'navy', linestyle = '-.')
    plt.title('Model loss', fontsize=18)
    plt.ylabel('Loss',fontsize=16)
    plt.xlabel('Epoch',fontsize=16)
    plt.legend(['Train', 'Test'], loc='upper right', fontsize=16)
    plt.show()
    plot_model(model, to_file='model.png', show_shapes=True)

    plt.plot(H.history['precision'], color = 'turquoise', label = "Train Precision")
    plt.plot(H.history['val_precision'], color = 'navy', label = "Test Precision")
    plt.plot(H.history['recall'],color = 'turquoise', linestyle = '--', label = 'Train Recall')
    plt.plot(H.history['recall'], color = 'navy', linestyle = '--', label = "Test Recall")
    plt.plot(H.history['fbeta'], color = 'turquoise', linestyle = ':', label = 'Train F1 Score')
    plt.plot(H.history['val_fbeta'], color = 'navy', linestyle= ':', label = "Test F1 Score")
    plt.title("Other model metrics: Threshold 0.5", fontsize = 18)
    plt.xlabel('Epoch', fontsize = 16)
    plt.legend(loc = 'lower right', fontsize = 8)
    plt.show()


    y_sum = np.sum(y_train_code, axis = 0)
    plt.bar(letters, y_sum, color = 'turquoise', label = "Train")
    plt.bar(letters, test_num, color = 'navy', label = "Test")
    plt.legend()
    plt.title("Letter frequency from y_train words")

    mean_y = np.mean(y_train_pred, axis = 0)
    plt.bar(letters, mean_y, color = 'navy')
    plt.title("Mean prediction from CNN across y_train corpus")
    preds_1 = pd.DataFrame(preds_y_test)



    for col, mean in zip(preds_1, mean_y):
        preds_1[col] = np.where(preds_1[col] >= mean+ 0.007, 1, 0)

    preds_num_1 = preds_1.values
    newdf = pd.DataFrame()
    newdf['letter'] = letters
    newdf['y_test_freq'] = np.sum(y_test_code, axis = 0)
    newdf['threshold']= np.round(mean_y + 0.007, 3)
    newdf['y_pred'] = np.sum(preds_1, axis = 0)

    # model.save('models/cnn_model_82acc_31919')
    preds_y_test = model.predict(X_test_lst)
    preds_bin = np.where(preds >= 0.40, 1, 0)
    CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyz "

    letters = [letter for letter in CHAR_VECTOR]
    y_df = pd.DataFrame(y_test_code, columns = letters)
    for col in y_df:
        y_df[col] = np.where(y_df[col]==1, col, 0)
    # y_df.replace({'0':np.nan}, inplace = True)

    y_pred_df = preds_1.copy()
    y_pred_df.columns = letters
    for col in y_pred_df:
        y_pred_df[col] = np.where(y_pred_df[col]==1, col, 0)

    # y_pred_df.replace({'0':np.nan}, inplace = True)
    y_df_ur = np.ravel(y_df.values)

    y_pred_df_ur = np.ravel(y_pred_df.values)

def standard_confusion_matrix(y_pred, y_true):
    cf_dict = {'TP':0, 'FP':0, 'TN':0,'FN':0}
    for i,j in zip(y_pred, y_true):
        if i == 1 and j == i:
            cf_dict['TP'] += 1
        elif i == 0 and j == i:
            cf_dict['TN'] += 1
        elif i == 1 and j == 0:
            cf_dict['FP'] += 1
        elif i == 0 and j == 1:
            cf_dict['FN'] +=1

    cf_mat = np.array([[cf_dict['TP'],cf_dict['FP']],[cf_dict['FN'], cf_dict['TN']]])

    return cf_mat, cf_dict

    cm = confusion_matrix(y_target = y_df_ur, y_predicted = y_pred_df_ur, binary = False)

    fig, ax = plot_confusion_matrix(conf_mat=cm, cmap = 'Blues', figsize = (10,10))


    recalls = []
    precisions = []
    accuracies = []

    for true, pred in zip(y_test_code.T, preds_num_1.T):
        cf_dict = {'TP':0, 'FP':0, 'TN':0,'FN':0}
        for i,j in zip(pred, true):
            if i == 1 and j == i:
                cf_dict['TP'] += 1
            elif i == 0 and j == i:
                cf_dict['TN'] += 1
            elif i == 1 and j == 0:
                cf_dict['FP'] += 1
            elif i == 0 and j == 1:
                cf_dict['FN'] +=1
        print(cf_dict)
        if cf_dict['TP'] + cf_dict['FN'] == 0:
            recall = cf_dict['TP'] / (cf_dict['TP'] + cf_dict['FN']+1)
        else:
            recall = cf_dict['TP'] / (cf_dict['TP'] + cf_dict['FN'])
        precision = cf_dict['TP']/ (cf_dict['TP'] + cf_dict['FP'])
        accuracy = (cf_dict['TP'] + cf_dict['TN'])/(cf_dict['TP']+cf_dict['TN']+cf_dict['FP']+cf_dict['FN'])
        recalls.append(recall)
        precisions.append(precision)
        accuracies.append(accuracy)

    newdf['accuracy'] = np.round(accuracies, 3)
    newdf['recall'] = np.round(recalls, 3)
    newdf['precision'] = np.round(precisions, 3)
