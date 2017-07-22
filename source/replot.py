from __future__ import print_function

from random import shuffle
import keras
import utility
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import tensorflow as tf
from PIL import Image
from keras import backend as K
import cv2

imgs = np.ones(shape=(8*274, 2740, 3))
for i in range(0, 8):
    name = './supervised/epoch_' + str(i*4 +1) + '.jpg'
    img = cv2.imread(name)
    imgs[i*274:(i*274+274), :, :] = img[(2*274):(3*274), :, :]

cv2.imwrite('3.jpg', imgs)