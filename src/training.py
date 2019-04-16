from keras import backend as K
from keras.optimizers import Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from Image_Generator import TextImageGenerator
from Model import get_Model
from parameter import *
K.set_learning_phase(0)

# # Model description and training

model = get_Model(training=True)

try:
    model.load_weights('LSTM+BN4--26--0.011.hdf5')
    print("...Previous weight data...")
except:
    print("...New weight data...")
    pass

train_file_path = 'data/pad_img/train/'
words_train = TextImageGenerator(train_file_path, img_w, img_h, batch_size, downsample_factor)
words_train.build_data()

valid_file_path = 'data/pad_img/test/'
words_val = TextImageGenerator(valid_file_path, img_w, img_h, val_batch_size, downsample_factor)
words_val.build_data()

ada = Adadelta()

early_stop = EarlyStopping(monitor='loss', min_delta=0.001, patience=4, mode='min', verbose=1)
checkpoint = ModelCheckpoint(filepath='LSTM+BN5--{epoch:02d}--{val_loss:.3f}.hdf5', monitor='loss', verbose=1, mode='min', period=1)
tensorboard = TensorBoard(log_dir='./logs', batch_size=100, write_graph=True, write_grads=True, write_images=True, histogram_freq=1)
# the loss calc occurs elsewhere, so use a dummy lambda func for the loss
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada, metrics = ['accuracy'])

# captures output of softmax so we can decode the output during visualization
H = model.fit_generator(generator=words_train.next_batch(),
                    steps_per_epoch=1000,
                    epochs=1000,
                    # callbacks=[checkpoint],
                    validation_data=words_val.next_batch(),
                    validation_steps=1000,
                    verbose = 1)
