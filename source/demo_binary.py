from __future__ import print_function

import keras
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from PIL import Image
from keras import backend as K
from keras import metrics

import model_ini

total = 1147
path = '/media/mjia/Data/SUN3D/demo2/'

def zero_mask(y):
    return tf.to_float(K.not_equal(K.zeros_like(y), y))

def zero_mask_inv(y):
    return tf.to_float(K.equal(K.zeros_like(y), y))

def my_loss(y_true, y_pred):
    return K.mean(tf.multiply(K.square(y_true - y_pred), zero_mask(y_true)))
    #return K.mean(tf.multiply(K.square(y_pred - y_true), tf.div(y_true, y_true)), axis=-1)

def metric_L1_real(y_true, y_pred):
    return K.mean(tf.realdiv(tf.multiply(K.abs(y_pred-y_true), zero_mask(y_true)), tf.add(y_true, zero_mask_inv(y_true))))

def metric_L1_inv(y_true, y_pred):
    return K.mean(K.abs(tf.realdiv(zero_mask(y_true), y_pred) - tf.realdiv(zero_mask(y_true), tf.add(y_true, zero_mask_inv(y_true)))))

def showImange(x, yy, depth, batchSize):
    img = np.empty(shape=(batchSize, 224, 960, 3))
    xx = np.empty(shape=(batchSize, 224, 320, 6))
    xx[:, :, :, :] = x[:, ::2, ::2, :]
    for i in range(batchSize):
        #imgx = Image.fromarray(x[i][:,:,0:3], 'RGB')
        #imgx.show()
        #imgy = Image.fromarray(30*np.float32(y[5][i][:,:,0]), 'F')
        #imgy.show()
        #imgd = Image.fromarray(30*depth[5][i][:, :, 0], 'F')
        #imgd.show()
        img[i][:,0:320,:] = np.float32(xx[i][:,:,3:6])
        img[i][:,320:640,0] = 30*np.float32(yy)
        img[i][:,320:640,1] = 30*np.float32(yy)
        img[i][:,320:640,2] = 30*np.float32(yy)
        img[i][:,640:960,0] = 30*depth[5][i][:,:,0]
        img[i][:,640:960,1] = 30*depth[5][i][:,:,0]
        img[i][:,640:960,2] = 30*depth[5][i][:,:,0]
        imgtoshow = Image.fromarray(np.uint8(img[i]), 'RGB')
        return imgtoshow

# initialize the model
img_rows, img_cols = 448, 640
input_shape = (img_rows, img_cols, 6)
model = model_ini.model_init_binary(input_shape)

model.compile(loss="categorical_crossentropy",
              metrics=[metrics.categorical_accuracy],
              optimizer=keras.optimizers.Adadelta())

model.load_weights('../../exp_data/trained_models/model_epoch_8.hdf5')
#loss = model.evaluate_generator(utility.data_generator(isTrain = False, isGAN= False, batchSize = 20), steps = 255)

x = np.empty(shape=(1, 448, 640, 6))
image_mean = np.zeros(shape=(448, 640, 6))
image_mean[:, :, 0] = 114 * np.ones(shape=(448, 640))
image_mean[:, :, 1] = 105 * np.ones(shape=(448, 640))
image_mean[:, :, 2] = 97 * np.ones(shape=(448, 640))
image_mean[:, :, 3] = 114 * np.ones(shape=(448, 640))
image_mean[:, :, 4] = 105 * np.ones(shape=(448, 640))
image_mean[:, :, 5] = 97 * np.ones(shape=(448, 640))
#my_video = cv2.VideoWriter(filename='video.avi', fourcc=cv2.VideoWriter_fourcc('M','J','P','G'), fps=10, frameSize=(224, 960), isColor=True)

#for i in range(1, 20):
#    x = np.empty(shape=(1, 448, 640, 6))
#    filename = '/media/mjia/Data/SUN3D/val/' + str(np.random.randint(1, 750)).zfill(7) + '.mat'
#    xx = sio.loadmat(filename)
#    x[:, :, :, 0:3] = xx['Data']['image'][0][0][0][0][16:464, :, :] - image_mean
#    x /= 255
#    depth = model.predict_on_batch(x)

#    img = xx['Data']['image'][0][0][0][1][16:464, :, :]
#    depth_gt = xx['Data']['depth'][0][0][0][1][16:464, :]
#    dict_to_save = {'img': img, 'depth': depth[5], 'depth_gt': depth_gt}
#    filename = './For_Yiming/model1_val/' + str(i).zfill(3) + '.mat'
#    sio.savemat(filename, dict_to_save)



for i in range(1, total+1):
    filename = path + str(i).zfill(7) + '.mat'
    xx = sio.loadmat(filename)
    yy = xx['Data']['depth'][0][0][0][0][16:464, :]
    y1 = yy[::2, ::2]
    if i == 1:
        x[0, :, :, 3:6] = xx['Data']['image'][0][0][0][0][16:464, :, :]
        continue
    else:
        #x[0, :, :, 0:3] = x[0, :, :, 3:6]  #binocular
        x[0, :, :, 0:3] = xx['Data']['image'][0][0][0][0][16:464, :, :]
        x[0, :, :, 3:6] = xx['Data']['image'][0][0][0][0][16:464, :, :]

    xRGB = x.astype('uint8');
    xPred = x.astype('float32') - image_mean

    xPred /= 255
    depth = model.predict_on_batch(xPred)
    image_to_show = showImange(xRGB, y1, depth, 1)
    #image_to_show.close()
    #image_to_show.show(title=1)
    plt.imshow(image_to_show)
    plt.pause(0.001)