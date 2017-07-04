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

# initialize the models
model_judge = model_ini.model_judgement(input_shape)

# compile the models
model_judge.compile(loss=keras.Loss.mean_squared_error,
                  metrics=[utility.metric_L1_real],
                  optimizer=keras.optimizers.Adadelta())


# load weight

########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################
########################################### main ##################################################################

loss = np.empty(shape=(40, 13))

for i in range(1, 40):

    history = model_judge.fit_generator(utility.judgement_generator(isTrain = True, batchSize = 10), steps_per_epoch = 4000, epochs = 1)
    loss[i] = model_judge.evaluate_generator(utility.judgement_generator(isTrain = False, batchSize = 20), steps = 250)
    filename = '../../exp_data/trained_models/model_epoch_' + str(i) + '.hdf5'
    model_judge.save_weights(filename)
    filename = '../../exp_data/trained_models/model_epoch_train' + str(i)
    np.save(filename, history.history)
    filename = '../../exp_data/trained_models/model_epoch_val' + str(i)
    np.save(filename, loss[i])

print('\n')
np.save('../../exp_data/trained_models/loss', loss)