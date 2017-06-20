from keras.models import Sequential
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D, UpSampling2D
from keras.layers import Input, Conv2DTranspose, concatenate, Activation, Dense
from keras.models import Model

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
    pr1p = Conv2D(filters=3, kernel_size=(3,3), strides=(1,1), padding="same")(iconv1)
    pr1 = Activation('softmax')(pr1p)

    model = Model(inputs=a, outputs=[pr6, pr5, pr4, pr3, pr2, pr1])

    return model
