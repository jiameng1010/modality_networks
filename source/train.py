from __future__ import print_function
from random import shuffle
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import scipy
import itertools
import numpy as np
import scipy.io as sio
from PIL import Image

train = 14409
val = 750
batchSize = 5

def showImange(x, y, depth, batchSize):
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
        img[i][:,320:640,0] = 30*np.float32(y[5][i][:,:,0])
        img[i][:,320:640,1] = 30*np.float32(y[5][i][:,:,0])
        img[i][:,320:640,2] = 30*np.float32(y[5][i][:,:,0])
        img[i][:,640:960,0] = 30*depth[5][i][:,:,0]
        img[i][:,640:960,1] = 30*depth[5][i][:,:,0]
        img[i][:,640:960,2] = 30*depth[5][i][:,:,0]
        imgtoshow = Image.fromarray(np.uint8(img[i]), 'RGB')
        imgtoshow.show()

        #imgd.close()
        #imgx.close()
        #imgy.close()



def train_epoch(model):
    index = [[i] for i in range(1,val)]
    shuffle(index)
    sum_loss = [0,0,0,0,0,0]

    for i in range(int(train/batchSize)-1):
        (xRGB, x,y) = loadData(index, i*batchSize, batchSize)
        depth = model.predict_on_batch(x);
        showImange(xRGB, y, depth, 5)

    print(index)
    return model




def loadData(index, index_begin, batchSize):
    x = np.empty(shape=(batchSize, 448, 640, 6))
    yy = np.empty(shape=(448, 640))
    y1 = np.empty(shape=(batchSize, 224, 320, 1))
    y2 = np.empty(shape=(batchSize, 112, 160, 1))
    y3 = np.empty(shape=(batchSize, 56, 80, 1))
    y4 = np.empty(shape=(batchSize, 28, 40, 1))
    y5 = np.empty(shape=(batchSize, 14, 20, 1))
    y6 = np.empty(shape=(batchSize, 7, 10, 1))
    for i in range(batchSize):
        number_of_file = str(index[index_begin+i][0])
        filename = '/media/mjia/Data/SUN3D/val/' + number_of_file.zfill(7) + '.mat'
        xx = sio.loadmat(filename)
        x[i,:,:,0:3] = xx['Data']['image'][0][0][0][0][16:464,:,:]
        x[i,:,:,3:6] = xx['Data']['image'][0][0][0][1][16:464,:,:]
        yy = xx['Data']['depth'][0][0][0][1][16:464,:]
        yy = yy.astype('float32')
        y1[i, :, :, 0] = yy[::2, ::2]
        y2[i, :, :, 0] = y1[i, ::2, ::2, 0]
        y3[i, :, :, 0] = y2[i, ::2, ::2, 0]
        y4[i, :, :, 0] = y3[i, ::2, ::2, 0]
        y5[i, :, :, 0] = y4[i, ::2, ::2, 0]
        y6[i, :, :, 0] = y5[i, ::2, ::2, 0]

    xRGB = x.astype('uint8');
    x = x.astype('float32')
    x /= 255
    y = [y6, y5, y4, y3, y2, y1]

    return (xRGB, x,y)





#index = [[i] for i in range(1,train)]
#shuffle(index)
#(x,y) = loadData(index, 500, batchSize)