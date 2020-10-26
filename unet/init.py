import os

import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Concatenate, concatenate, BatchNormalization, Activation, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img, save_img

import random
import numpy as np
from matplotlib import pyplot as plt

try:
  from google.colab import drive
  drive.mount('/content/gdrive')
  data_path = '/content/gdrive/My Drive/Colab Notebooks/martin/fluo'
except:
  data_path = '/home/stefano/Scuola/tud/_classes/3/rp/martin/attila/data/fluo'

image_size = (128, 128)
input_shape = (*image_size, 1)
verbose = 1
batch_size = 32
epochs = 100
best_model_weights = 'model.h5'
