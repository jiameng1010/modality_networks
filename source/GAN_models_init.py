from keras.models import Sequential
from keras.layers import ZeroPadding2D
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense
from keras.layers import Input, Conv2DTranspose, concatenate, Flatten
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
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu', name='G_C_1')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='G_C_2')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='G_C_3')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_4')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='G_C_5')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_6')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='G_C_7')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_8')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='G_C_9')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_10')(conv6a)

    upconv5 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same", name='G_6_1')(conv6b)
    pr6 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_6_2')(conv6b)
    pr6up = UpSampling2D(size=(2,2), name='G_6_3')(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3, name='G_6_4')

    iconv5 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_5_1')(inter5)
    pr5 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_5_2')(iconv5)
    pr5up = UpSampling2D(size=(2,2), name='G_5_3')(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_5_4')(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3, name='G_5_5')

    iconv4 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_4_1')(inter4)
    pr4 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_4_2')(iconv4)
    pr4up = UpSampling2D(size=(2,2), name='G_4_3')(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_4_4')(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3, name='G_4_5')

    iconv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_3_1')(inter3)
    pr3 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_3_2')(iconv3)
    pr3up = UpSampling2D(size=(2,2), name='G_3_3')(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_3_4')(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3, name='G_3_5')

    iconv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_2_1')(inter2)
    pr2 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_2_2')(iconv2)
    pr2up = UpSampling2D(size=(2,2), name='G_2_3')(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_2_4')(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3, name='G_2_5')

    iconv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_1_1')(inter1)
    pr1 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_1_2')(iconv1)

    ########################## Discriminators ######################################
    ########################## Discriminators ######################################
    ########################## Discriminators ######################################
    ########################## Discriminators ######################################
    ########################## Discriminators ######################################
    ########################## Discriminators ######################################
    ########################## Discriminators ######################################
    ########################## Discriminators ######################################
    ########################## Discriminators ######################################

    d1_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_1_1')(pr1)
    d1_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_2')(d1_conv1)
    d1_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_1_3')(d1_conv2)
    d1_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_4')(d1_conv3)
    d1_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_1_5')(d1_conv4)
    d1_conv6 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_6')(d1_conv5)
    d1_conv7 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_7')(d1_conv6)
    d1_conv8 = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_8')(d1_conv7)
    d1_conv8_f = Flatten(name='D_1_9')(d1_conv8)
    d1_dens9 = Dense(units=512, activation='relu', name='D_1_10')(d1_conv8_f)
    d1_dens10 = Dense(units=32, activation='softmax', name='D_1_11')(d1_dens9)

    d2_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_2_1')(pr2)
    d2_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_2')(d2_conv1)
    d2_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_2_3')(d2_conv2)
    d2_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_4')(d2_conv3)
    d2_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_2_5')(d2_conv4)
    d2_conv6 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_6')(d2_conv5)
    d2_conv7 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_7')(d2_conv6)
    d2_conv7_f = Flatten(name='D_2_8')(d2_conv7)
    d2_dens8 = Dense(units=512, activation='relu', name='D_2_9')(d2_conv7_f)
    d2_dens9 = Dense(units=32, activation='relu', name='D_2_10')(d2_dens8)

    d3_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_3_1')(pr3)
    d3_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_3_2')(d3_conv1)
    d3_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_3_3')(d3_conv2)
    d3_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_3_4')(d3_conv3)
    d3_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_3_5')(d3_conv4)
    d3_conv6 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_3_6')(d3_conv5)
    d3_conv6_f = Flatten(name='D_3_7')(d3_conv6)
    d3_dens7 = Dense(units=256, activation='relu', name='D_3_8')(d3_conv6_f)
    d3_dens8 = Dense(units=32, activation='relu', name='D_3_9')(d3_dens7)

    d4_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_4_1')(pr4)
    d4_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_4_2')(d4_conv1)
    d4_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_4_3')(d4_conv2)
    d4_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_4_4')(d4_conv3)
    d4_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_4_5')(d4_conv4)
    d4_conv5_f = Flatten(name='D_4_6')(d4_conv5)
    d4_dens6 = Dense(units=256, activation='relu', name='D_4_7')(d4_conv5_f)
    d4_dens7 = Dense(units=32, activation='relu', name='D_4_8')(d4_dens6)

    d5_conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_5_1')(pr5)
    d5_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_5_2')(d5_conv1)
    d5_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_5_3')(d5_conv2)
    d5_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_5_4')(d5_conv3)
    d5_conv4_f = Flatten(name='D_5_5')(d5_conv4)
    d5_dens5 = Dense(units=128, activation='relu', name='D_5_6')(d5_conv4_f)
    d5_dens6 = Dense(units=32, activation='relu', name='D_5_7')(d5_dens5)

    d6_conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_6_1')(pr6)
    d6_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_6_2')(d6_conv1)
    d6_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_6_3')(d6_conv2)
    d6_conv3_f = Flatten(name='D_6_4')(d6_conv3)
    d6_dens4 = Dense(units=128, activation='relu', name='D_6_5')(d6_conv3_f)
    d6_dens5 = Dense(units=32, activation='relu', name='D_6_6')(d6_dens4)

    d_merge = concatenate([d6_dens5, d5_dens6, d4_dens7, d3_dens8, d2_dens9, d1_dens10], name='D_M_1')
    d_merge_1 = Dense(units=32, activation='relu', name='D_M_2')(d_merge)
    d = Dense(units=2, activation='softmax', name='D_M_3')(d_merge_1)

    model = Model(inputs=a, outputs=[pr6, pr5, pr4, pr3, pr2, pr1, d])

    return model

def model_g(input_shape):
    a = Input(shape=input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7), strides=(2,2), padding="same", activation='relu', name='G_C_1')(a)
    conv2 = Conv2D(filters=128, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='G_C_2')(conv1)
    conv3a = Conv2D(filters=256, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='G_C_3')(conv2)
    conv3b = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_4')(conv3a)
    conv4a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='G_C_5')(conv3b)
    conv4b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_6')(conv4a)
    conv5a = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='G_C_7')(conv4b)
    conv5b = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_8')(conv5a)
    conv6a = Conv2D(filters=1024, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='G_C_9')(conv5b)
    conv6b = Conv2D(filters=1024, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_C_10')(conv6a)

    upconv5 = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), input_shape=(7, 10, 1024), padding="same", name='G_6_1')(conv6b)
    pr6 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_6_2')(conv6b)
    pr6up = UpSampling2D(size=(2,2), name='G_6_3')(pr6)
    inter5 = concatenate([upconv5, conv5b, pr6up], axis=3, name='G_6_4')

    iconv5 = Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_5_1')(inter5)
    pr5 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_5_2')(iconv5)
    pr5up = UpSampling2D(size=(2,2), name='G_5_3')(pr5)
    upconv4 = Conv2DTranspose(filters=256, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_5_4')(iconv5)
    inter4 = concatenate([upconv4, conv4b, pr5up], axis=3, name='G_5_5')

    iconv4 = Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_4_1')(inter4)
    pr4 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_4_2')(iconv4)
    pr4up = UpSampling2D(size=(2,2), name='G_4_3')(pr4)
    upconv3 = Conv2DTranspose(filters=128, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_4_4')(iconv4)
    inter3 = concatenate([upconv3, conv3b, pr4up], axis=3, name='G_4_5')

    iconv3 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_3_1')(inter3)
    pr3 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_3_2')(iconv3)
    pr3up = UpSampling2D(size=(2,2), name='G_3_3')(pr3)
    upconv2 = Conv2DTranspose(filters=64, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_3_4')(iconv3)
    inter2 = concatenate([upconv2, conv2, pr3up], axis=3, name='G_3_5')

    iconv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_2_1')(inter2)
    pr2 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_2_2')(iconv2)
    pr2up = UpSampling2D(size=(2,2), name='G_2_3')(pr2)
    upconv1 = Conv2DTranspose(filters=32, kernel_size=(4,4), strides=(2,2), dilation_rate=(2,2), padding="same", name='G_2_4')(iconv2)
    inter1 = concatenate([upconv1, conv1, pr2up], axis=3, name='G_2_5')

    iconv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='G_1_1')(inter1)
    pr1 = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding="same", name='G_1_2')(iconv1)

    model = Model(inputs=a, outputs=[pr6, pr5, pr4, pr3, pr2, pr1])

    return model

def model_d():
    pr1 = Input(shape=(224, 320, 1))
    d1_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_1_1')(pr1)
    d1_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_2')(d1_conv1)
    d1_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_1_3')(d1_conv2)
    d1_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_4')(d1_conv3)
    d1_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_1_5')(d1_conv4)
    d1_conv6 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_6')(d1_conv5)
    d1_conv7 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_7')(d1_conv6)
    d1_conv8 = Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_1_8')(d1_conv7)
    d1_conv8_f = Flatten(name='D_1_9')(d1_conv8)
    d1_dens9 = Dense(units=512, activation='relu', name='D_1_10')(d1_conv8_f)
    d1_dens10 = Dense(units=32, activation='softmax', name='D_1_11')(d1_dens9)

    pr2 = Input(shape=(112, 160, 1))
    d2_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_2_1')(pr2)
    d2_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_2')(d2_conv1)
    d2_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_2_3')(d2_conv2)
    d2_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_4')(d2_conv3)
    d2_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_2_5')(d2_conv4)
    d2_conv6 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_6')(d2_conv5)
    d2_conv7 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_2_7')(d2_conv6)
    d2_conv7_f = Flatten(name='D_2_8')(d2_conv7)
    d2_dens8 = Dense(units=512, activation='relu', name='D_2_9')(d2_conv7_f)
    d2_dens9 = Dense(units=32, activation='relu', name='D_2_10')(d2_dens8)

    pr3 = Input(shape=(56, 80, 1))
    d3_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_3_1')(pr3)
    d3_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_3_2')(d3_conv1)
    d3_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_3_3')(d3_conv2)
    d3_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_3_4')(d3_conv3)
    d3_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_3_5')(d3_conv4)
    d3_conv6 = Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_3_6')(d3_conv5)
    d3_conv6_f = Flatten(name='D_3_7')(d3_conv6)
    d3_dens7 = Dense(units=256, activation='relu', name='D_3_8')(d3_conv6_f)
    d3_dens8 = Dense(units=32, activation='relu', name='D_3_9')(d3_dens7)

    pr4 = Input(shape=(28, 40, 1))
    d4_conv1 = Conv2D(filters=32, kernel_size=(5,5), strides=(2,2), padding="same", activation='relu', name='D_4_1')(pr4)
    d4_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_4_2')(d4_conv1)
    d4_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_4_3')(d4_conv2)
    d4_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_4_4')(d4_conv3)
    d4_conv5 = Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_4_5')(d4_conv4)
    d4_conv5_f = Flatten(name='D_4_6')(d4_conv5)
    d4_dens6 = Dense(units=256, activation='relu', name='D_4_7')(d4_conv5_f)
    d4_dens7 = Dense(units=32, activation='relu', name='D_4_8')(d4_dens6)

    pr5 = Input(shape=(14, 20, 1))
    d5_conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_5_1')(pr5)
    d5_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_5_2')(d5_conv1)
    d5_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_5_3')(d5_conv2)
    d5_conv4 = Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_5_4')(d5_conv3)
    d5_conv4_f = Flatten(name='D_5_5')(d5_conv4)
    d5_dens5 = Dense(units=128, activation='relu', name='D_5_6')(d5_conv4_f)
    d5_dens6 = Dense(units=32, activation='relu', name='D_5_7')(d5_dens5)

    pr6 = Input(shape=(7, 10, 1))
    d6_conv1 = Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_6_1')(pr6)
    d6_conv2 = Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', name='D_6_2')(d6_conv1)
    d6_conv3 = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding="same", activation='relu', name='D_6_3')(d6_conv2)
    d6_conv3_f = Flatten(name='D_6_4')(d6_conv3)
    d6_dens4 = Dense(units=128, activation='relu', name='D_6_5')(d6_conv3_f)
    d6_dens5 = Dense(units=32, activation='relu', name='D_6_6')(d6_dens4)

    d_merge = concatenate([d6_dens5, d5_dens6, d4_dens7, d3_dens8, d2_dens9, d1_dens10], name='D_M_1')
    d_merge_1 = Dense(units=32, activation='relu', name='D_M_2')(d_merge)
    d = Dense(units=2, activation='softmax', name='D_M_3')(d_merge_1)

    model = Model(input=[pr6, pr5, pr4, pr3, pr2, pr1], outputs=d)
    return model