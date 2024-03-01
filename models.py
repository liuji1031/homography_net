import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.convnext import ConvNeXtBase, preprocess_input

from keras.layers import BatchNormalization as BN
from keras.layers import Conv2D, MaxPool2D,  \
    Dropout, Dense, Input, concatenate,      \
    GlobalAveragePooling2D, AveragePooling2D,\
    Flatten

def backbone_VGG19():
    """backbone choice 1: VGG19

    Returns:
        _type_: _description_
    """
    base = VGG19(
        include_top=False,
        weights='imagenet',
        input_shape=(128,128,3),
        pooling='max'
    )
    base.trainable = False

    # retrieve only some part of the base
    input = base.input

    # use earlier conv output
    # output = base.get_layer('block2_conv2').output
    output = base.get_layer('block3_conv4').output

    base_ = keras.Model(inputs=input, outputs=output)
    base_.trainable = False

    return base_

def backbone_inception():
    base = InceptionV3( include_top=False,
                        weights='imagenet',
                        input_shape=(128,128,3),
                        pooling=None,)
    base.trainable = False
    base.summary()

    # retrieve only some part of the base
    input = base.input
    output = base.get_layer('activation_4').output

    base_ = keras.Model(inputs=input, outputs=output)
    base_.trainable = False
    
    return base_

def model_v1(verbose=False):
    # VGG19 backbone
    base = backbone_VGG19()

    im_shape = (128,128,3)
    input1 = keras.layers.Input(shape=im_shape)
    input2 = keras.layers.Input(shape=im_shape)

    x1 = preprocess_input(input1)
    x2 = preprocess_input(input2)

    x1 = base(x1)
    x2 = base(x2)
    x = keras.layers.Concatenate(axis=-1)((x1,x2)) # 32 by 32 by 512

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

    if verbose:
        model.summary()

    return model

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv2D(filters_1x1, (1, 1),
                      padding='same',
                      activation='relu',
                      )(x)
    
    conv_3x3 = Conv2D(filters_3x3_reduce,(1, 1),
                      padding='same',
                      activation='relu',
                      )(x)
    
    conv_3x3 = Conv2D(filters_3x3,(3, 3),
                      padding='same',
                      activation='relu',
                      )(conv_3x3)

    conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1),
                      padding='same',
                      activation='relu',
                      )(x)
    
    conv_5x5 = Conv2D(filters_5x5, (5, 5),
                      padding='same',
                      activation='relu',
                      )(conv_5x5)

    pool_proj = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1),
                       padding='same',
                       activation='relu',
                      )(pool_proj)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj],
                         axis=3, name=name)
    
    return output

def model_v2(verbose=False):

    base = backbone_VGG19()

    im_shape = (128,128,3)
    input1 = keras.layers.Input(shape=im_shape)
    input2 = keras.layers.Input(shape=im_shape)

    x1 = preprocess_input(input1)
    x2 = preprocess_input(input2)

    x1 = base(x1)
    x2 = base(x2)
    x = keras.layers.Concatenate(axis=-1)((x1,x2)) # 32 by 32 by 512

    config1 = dict(filters_1x1=64,
                filters_3x3_reduce=96,
                filters_3x3=128,
                filters_5x5_reduce=16,
                filters_5x5=32,
                filters_pool_proj=32)

    x = inception_module(x,name='inception_1',**config1)
    x = BN()(x)

    x = inception_module(x,name='inception_2',**config1)
    x = BN()(x)

    x = inception_module(x,name='inception_3',**config1)
    x = BN()(x)

    x = inception_module(x,name='inception_4',**config1)
    x = BN()(x)

    x = inception_module(x,name='inception_5',**config1)
    x = BN()(x)

    x = inception_module(x,name='inception_6',**config1)
    x = BN()(x)

    x = inception_module(x,name='inception_7',**config1)
    x = BN()(x)

    x = keras.layers.MaxPool2D(pool_size=(2,2),strides=2)(x)
    x = Conv2D(256, (1, 1),
                      padding='same',
                      activation='relu')(x)

    x = BN()(x)

    x = keras.layers.Flatten()(x)
    x = BN()(x)

    x = keras.layers.Dense(units=512, activation='relu',
                           kernel_initializer=keras.initializers.random_normal(stddev=0.01)
                           )(x)
    x = BN()(x)
    print("new init")
    x = keras.layers.Dense(units=512, activation='relu',
                           kernel_initializer=keras.initializers.random_normal(stddev=0.01)
                           )(x)
    x = BN()(x)

    output = keras.layers.Dense(units=8,activation=None,
                                kernel_initializer=keras.initializers.random_normal(stddev=0.01))(x)

    model = keras.Model(inputs=[input1,input2],outputs=output)

    if verbose:
        model.summary()

    return model