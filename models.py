import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import BatchNormalization as BN


def model_v1():

    base = VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(128,128,3),
        pooling='max'
    )
    base.trainable = False

    # retrieve only some part of the base
    input = base.input
    output = base.get_layer('block3_conv4').output

    base2 = keras.Model(inputs=input, outputs=output)
    base2.trainable = False
    # base2.summary()

    im_shape = (128,128,3)
    input1 = keras.layers.Input(shape=im_shape)
    input2 = keras.layers.Input(shape=im_shape)

    x1 = preprocess_input(input1)
    x2 = preprocess_input(input2)

    x1 = base2(x1)
    x2 = base2(x2)

    x = keras.layers.Concatenate()((x1,x2)) # 32 by 32 by 512

    x = keras.layers.Conv2D(filters=256,
                                kernel_size=(3,3),
                                activation='relu',
                                padding='same')(x)
    x = BN()(x)

    x = keras.layers.Conv2D(filters=256,
                                kernel_size=(3,3),
                                activation='relu',
                                padding='same')(x)
    x = BN()(x)

    x = keras.layers.Conv2D(filters=256,
                                kernel_size=(3,3),
                                activation='relu',
                                padding='same')(x)
    x = BN()(x)

    x = keras.layers.Conv2D(filters=256,
                                kernel_size=(3,3),
                                activation='relu',
                                padding='same')(x)
    x = BN()(x)

    # add some 3x3 conv
    x = keras.layers.Conv2D(filters=256,
                                kernel_size=(3,3),
                                strides=(2,2),
                                activation='relu',
                                padding='same')(x)

    x = BN()(x)

    x = keras.layers.Conv2D(filters=256,
                                kernel_size=(3,3),
                                strides=(2,2),
                                activation='relu',
                                padding='same')(x)

    x = BN()(x)
    # # add 1x1 conv
    # x = tf.keras.layers.Conv2D(filters=256,
    #                             kernel_size=(1,1),
    #                             activation='relu')(x)

    x = keras.layers.Flatten()(x)
    x = BN()(x)
    # x = tf.keras.layers.LayerNormalization()(x)

    x = keras.layers.Dense(units=512, activation='relu')(x)
    x = BN()(x)
    x = keras.layers.Dense(units=512, activation='relu')(x)
    x = BN()(x)
    output = keras.layers.Dense(units=8,activation=None)(x)

    model = keras.Model(inputs=[input1,input2],outputs=output)
    model.summary()

    return model