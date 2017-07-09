from keras.models import Sequential
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D, UpSampling2D, multiply, core
from keras.layers import Input, Conv2DTranspose, concatenate, Activation, Dense
from keras.layers.merge import Multiply, Add
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
import tensorflow as tf

def model_init(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    upconv5 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(conv6b)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1)

    model = Model(inputs=a, outputs=[pr6, pr5, pr4, pr3, pr2, pr1])

    return model

def discriminator(input_shape):

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)
    pr1 = Dense(256, activation='relu')(conv6b)
    pr2 = Dense(1, activation='softmax')(conv6b)

    model = Model(input=a, outputs=pr2)

    return model

def model_init_binary(input_shape):

    #model = Sequential()
    #model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    #model.add(Conv2D(64, (3, 3), activation='relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.25))
    #model.add(Flatten())
    #model.add(Dense(128, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(10, activation='softmax'))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(conv6a)

    upconv5 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(conv6b)
    #pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2,2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv5)
    #pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2,2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv4)
    #pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2,2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv3)
    #pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2,2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv2)
    #pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2,2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same")(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same", activation='softmax')(iconv1)
    #pr1b = Activation(K.softmax)(pr1)


    model = Model(inputs=a, outputs=[pr6, pr5, pr4, pr3, pr2, pr1])

    return model

def model_judgement(input_shape):
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    close = Input(shape=(224, 320, 1))
    far = Input(shape=(224, 320, 1))

    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv6a)

    upconv5 = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2),
                              input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(conv6b)
    # pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2, 2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv5)
    # pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2, 2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv4)
    # pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2, 2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv3)
    # pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2, 2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv2)
    # pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2, 2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv1)
    # pr1b = Activation(K.softmax)(pr1)

    far_w = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_w = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)

    far_ww = Multiply()([far_w, far])
    close_ww = Multiply()([close_w, close])
    pre = Add()([far_ww, close_ww])
    #pre = core.Lambda(judgement_merge(pr1[:,:,:,1], far, pr1[:,:,:,2], close), output_shape=(1,))
    #pre = MyLayer()([pr1[:,:,:,1], far, pr1[:,:,:,2], close])

    model = Model(inputs=[a, far, close], outputs=pre)

    return model



def model_judgement2():
    # model = Sequential()
    # model.add(Conv2D(32, kernel_size=(3, 3),
    #                 activation='relu',
    #                 input_shape=input_shape))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Flatten())
    # model.add(Dense(128, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(10, activation='softmax'))
    close1 = Input(shape=(224, 320, 1))
    far1 = Input(shape=(224, 320, 1))
    close2 = Input(shape=(112, 160, 1))
    far2 = Input(shape=(112, 160, 1))
    close3 = Input(shape=(56, 80, 1))
    far3 = Input(shape=(56, 80, 1))
    close4 = Input(shape=(28, 40, 1))
    far4 = Input(shape=(28, 40, 1))
    close5 = Input(shape=(14, 20, 1))
    far5 = Input(shape=(14, 20, 1))
    close6 = Input(shape=(7, 10, 1))
    far6 = Input(shape=(7, 10, 1))

    a = Input(shape=(448, 640, 6))
    conv1 = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), padding="same", activation='relu')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5, 5), strides=(2, 2), padding="same", activation='relu')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding="same", activation='relu')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(conv6a)

    upconv5 = Conv2DTranspose(filters=512, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2),
                              input_shape=(7, 10, 1024), padding="same")(conv6b)
    pr6 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(conv6b)
    pr6b = Activation(K.softmax)(pr6)
    pr6up = UpSampling2D(size=(2, 2))(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3)

    iconv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter5)
    pr5 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv5)
    pr5b = Activation(K.softmax)(pr5)
    pr5up = UpSampling2D(size=(2, 2))(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3)

    iconv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter4)
    pr4 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv4)
    pr4b = Activation(K.softmax)(pr4)
    pr4up = UpSampling2D(size=(2, 2))(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3)

    iconv3 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter3)
    pr3 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv3)
    pr3b = Activation(K.softmax)(pr3)
    pr3up = UpSampling2D(size=(2, 2))(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3)

    iconv2 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter2)
    pr2 = Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding="same")(iconv2)
    pr2b = Activation(K.softmax)(pr2)
    pr2up = UpSampling2D(size=(2, 2))(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4, 4), strides=(2, 2), dilation_rate=(2, 2), padding="same")(
        iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3)

    iconv1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='relu')(inter1)
    pr1 = Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding="same", activation='softmax')(iconv1)
    # pr1b = Activation(K.softmax)(pr1)

    far_1 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr1)
    close_1 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr1)
    far_2 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr2b)
    close_2 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr2b)
    far_3 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr3b)
    close_3 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr3b)
    far_4 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr4b)
    close_4 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr4b)
    far_5 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr5b)
    close_5 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr5b)
    far_6 = core.Lambda(lambda x: x[:, :, :, 0:1])(pr6b)
    close_6 = core.Lambda(lambda x: x[:, :, :, 1:2])(pr6b)

    far_p1 = Multiply()([far_1, far1])
    close_p1 = Multiply()([close_1, close1])
    pre1 = Add()([far_p1, close_p1])

    far_p2 = Multiply()([far_2, far2])
    close_p2 = Multiply()([close_2, close2])
    pre2 = Add()([far_p2, close_p2])

    far_p3 = Multiply()([far_3, far3])
    close_p3 = Multiply()([close_3, close3])
    pre3 = Add()([far_p3, close_p3])

    far_p4 = Multiply()([far_4, far4])
    close_p4 = Multiply()([close_4, close4])
    pre4 = Add()([far_p4, close_p4])

    far_p5 = Multiply()([far_5, far5])
    close_p5 = Multiply()([close_5, close5])
    pre5 = Add()([far_p5, close_p5])

    far_p6 = Multiply()([far_6, far6])
    close_p6 = Multiply()([close_6, close6])
    pre6 = Add()([far_p6, close_p6])

    model = Model(inputs=[a, close6, close5, close4, close3, close2, close1, far6, far5, far4, far3, far2, far1],
                  outputs=(pre6, pre5, pre4, pre3, pre2, pre1))

    return model


def model_overall(model_c, model_f, model_j):
    model_input = Input(shape=(448, 640, 6))

    far = model_f(model_input)
    close = model_c(model_input)

    model_output = model_j([model_input,
                                    close[0], close[1], close[2], close[3], close[4], close[5],
                                    far[0], far[1], far[2], far[3], far[4], far[5]])

    model = Model(inputs=model_input, outputs=model_output)

    return model




