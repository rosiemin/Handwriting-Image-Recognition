from keras import backend as K


# params
MAX_LEN_TEXT = 20
IMAGE_SIZE = (128, 32)
IMG_W, IMG_H = IMAGE_SIZE
NO_CHANNELS = 1

if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (NO_CHANNELS, IMG_W, IMG_H)
else:
    INPUT_SHAPE = (IMG_W, IMG_H, NO_CHANNELS)

BATCH_SIZE = 128
CONV_FILTERS = 16
KERNEL_SIZE = (3, 3)
POOL_SIZE = 2
DOWNSAMPLE_FACTOR = POOL_SIZE ** 2
TIME_DENSE_SIZE = 32
RNN_SIZE = 512
NO_OUTPUTS = 28
NO_EPOCHS = 5


# paths
WORDS_DATA = 'data/words.txt'
WORDS_TRAIN = 'trainset.txt'
WORDS_TEST = 'testset.txt'
CONFIG_MODEL = 'models/config/model_1.json'
WEIGHT_MODEL = 'models/model_1.pkl'
MODEL_CHECKPOINT = 'models/checkpoints/'
LOGGING = 'logs/'

# naming
WORDS_FOLDER = "data/words/"

"""
data
├── words
│   ├── a01
│   ├── a02
│   ├── a03
│   ├── a04
│   ├── a05
...
"""
