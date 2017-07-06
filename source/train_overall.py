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

# input image dimensions
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 6)

# initialize the models
model_far = model_ini.model_init(input_shape)
model_close = model_ini.model_init(input_shape)
model_judge = model_ini.model_judgement(input_shape)
model_overall = model_ini.model_overall(model_close, model_far, model_judge)

# load pre-trained
model_close.load_weights('./trained_models/model_close.hdf5')
model_far.load_weights('./trained_models/model_far.hdf5')
model_judge.load_weights('./trained_models/model_judge.hdf5')

# compile the models
model_overall.compile(loss=utility.my_loss,
                  metrics=[utility.metric_L1_real],
                  optimizer=keras.optimizers.Adadelta())

loss = np.empty(shape=(40, 13))

for i in range(1, 40):

    history = model_overall.fit_generator(utility.overall_generator(isTrain = True, batchSize = 5), steps_per_epoch = 4000, epochs = 1)
    loss[i] = model_overall.evaluate_generator(utility.overall_generator(isTrain = False, batchSize = 10), steps = 250)
    filename = '../../exp_data/trained_models/model_epoch_' + str(i) + '.hdf5'
    model_overall.save_weights(filename)
    filename = '../../exp_data/trained_models/model_epoch_train' + str(i)
    np.save(filename, history.history)
    filename = '../../exp_data/trained_models/model_epoch_val' + str(i)
    np.save(filename, loss[i])

print('\n')



