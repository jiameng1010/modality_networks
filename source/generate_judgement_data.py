from random import shuffle

import cv2
import keras
import numpy as np
import scipy.io as sio
import tensorflow as tf
import utility
from keras import backend as K
from keras import metrics
import model_ini

trainn = 102899
val = 5287
train_path = '/media/mjia/Data/SUN3D/train/'
val_path = '/media/mjia/Data/SUN3D/val/'

# input image dimensions
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 6)
depth_shape = (img_rows, img_cols, 1)

# initialize the models
model_close = model_ini.model_init(input_shape)
model_far = model_ini.model_init(input_shape)
#model_judge = model_ini.model_judgement(input_shape)

# compile the models
model_close.compile(loss=utility.my_loss,
                  metrics=[utility.metric_L1_real],
                  optimizer=keras.optimizers.Adadelta())
model_far.compile(loss=utility.my_loss,
                  metrics=[utility.metric_L1_real],
                  optimizer=keras.optimizers.Adadelta())


# load weight
model_close.load_weights('./trained_models/model_close.hdf5')
model_far.load_weights('./trained_models/model_far.hdf5')




########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
image_mean = np.zeros(shape=(448, 640, 3))
image_mean[:,:,0] = 114*np.ones(shape=(448, 640))
image_mean[:,:,1] = 105*np.ones(shape=(448, 640))
image_mean[:,:,2] = 97*np.ones(shape=(448, 640))
path = val_path
index = [[i] for i in range(1, trainn)]
i = 0
while(True):
    (x, y) = utility.loadDataGAN(index, i, 1, path, image_mean)
    i = i+1
    pre_far = model_far.predict_on_batch(x)
    pre_close = model_close.predict_on_batch(x)
    filename = '/media/mjia/Data/SUN3D/val_pr/' + str(i).zfill(7)
    np.save(filename, [pre_far, pre_close])
    print(i)