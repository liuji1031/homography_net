import tensorflow as tf
import keras
from keras.applications.vgg19 import VGG19, preprocess_input
from keras.layers import BatchNormalization as BN
from keras.layers import Conv2D, MaxPool2D,concatenate
from model.tensor_dlt import TensorDLT
from model.spatial_transformer import spatial_transformer_network

def get_model(mode="supervised"):

    if mode=="supervised":
        return homography_net_model()
    elif mode == "unsupervised":
        return get_full_model()

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
    inputs = base.input

    # use earlier conv output
    # output = base.get_layer('block2_conv2').output
    output = base.get_layer('block3_conv4').output

    base_ = keras.Model(inputs=inputs, outputs=output)
    base_.trainable = False

    return base_

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

def homography_net_model(verbose=False):

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
                           kernel_initializer=\
                           keras.initializers.random_normal(stddev=0.01)
                           )(x)
    x = BN()(x)
    print("new init")
    x = keras.layers.Dense(units=512, activation='relu',
                           kernel_initializer=\
                           keras.initializers.random_normal(stddev=0.01)
                           )(x)
    x = BN()(x)

    output = keras.layers.Dense(units=8,activation=None,
                                kernel_initializer=\
                                keras.initializers.random_normal(stddev=0.01)
                                )(x)

    model = keras.Model(inputs=[input1,input2],outputs=output)

    if verbose:
        model.summary()

    return model


def get_full_model(rho=32,
                   batch_size=8,
                   im_crop_shape=(128,128,3),
                   im_ori_shape=(240,320,3)
                   ):

    homography_net = homography_net_model()

    # retrieve the input of the network
    cropped_img1 = keras.layers.Input(shape=im_crop_shape)
    cropped_img2 = keras.layers.Input(shape=im_crop_shape)
    img_ori = keras.layers.Input(shape=im_ori_shape)

    h4pt_batch = homography_net([cropped_img1,cropped_img2]) # batch by 8
    h4pt_batch = tf.clip_by_value(h4pt_batch,clip_value_min=-rho,clip_value_max=rho)

    upper_left_corner = keras.layers.Input(shape=(2,)) # batch by 2

    # feed the homography net output to the TensorDLT to recover actual homography
    homography = TensorDLT(h4pt_batch=h4pt_batch,
                        upper_left_corner=upper_left_corner,
                        batch_size=batch_size)

    # use spatial transformer to get predicted image
    img_pred = spatial_transformer_network(img_ori,
                                        homography,
                                        img_height=im_ori_shape[0],
                                        img_width=im_ori_shape[1],
                                        )
    
    full_model = FullModel(inputs=[cropped_img1,
                                 cropped_img2,
                                 img_ori,
                                 upper_left_corner,
                                 ], 
                        outputs=[img_pred,h4pt_batch])

    # rename output layers
    full_model.layers[2]._name = 'h4pt_output'
    full_model.layers[-1]._name = 'img_output'

    return full_model

# pylint: disable=not-callable
class FullModel(keras.Model):
    """define a wrapper model so we can custom what happens when we call
    model.fit

    Args:
        keras (_type_): _description_
    """
    def __init__(self, inputs, outputs):
        super().__init__(inputs=inputs, outputs=outputs)
        self.loss_tracker = keras.metrics.MeanAbsoluteError(name='mae_loss')
        self.metric_h4pt = keras.metrics.MeanAbsoluteError(name='mae_h4pt')
        self.loss_tracker_val = \
            keras.metrics.MeanAbsoluteError(name='val_mae_loss')
        self.metric_h4pt_val = \
            keras.metrics.MeanAbsoluteError(name='val_mae_h4pt')

    # @tf.function
    def train_step(self, data):
        data_in, data_out = data
        with tf.GradientTape() as tape:
            model_out = self(data_in, training=True)
            loss = tf.reduce_mean(keras.losses.mean_absolute_error(
                                                    y_true=data_out[0],
                                                    y_pred=model_out[0]))

        grads = tape.gradient(loss, self.trainable_variables)

        # check if grad has nan, if so simply return
        skip = False
        for g in grads:
            try:
                tf.debugging.check_numerics(g, message='Checking grad')
            except Exception as e:
                tf.print("==================== nan found ====================",)
                skip = True
                break

        if skip is False:
            # do gradient update
            self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

            self.loss_tracker.update_state(data_out[0], model_out[0])
            self.metric_h4pt.update_state(data_out[-1], model_out[-1])

        return {"mae_loss": self.loss_tracker.result(),
            "mae_h4pt": self.metric_h4pt.result()}
    
    def test_step(self, data):
        data_in, data_out = data
        model_out = self(data_in, training=False)

        self.loss_tracker_val.update_state(data_out[0], model_out[0])
        self.metric_h4pt_val.update_state(data_out[-1], model_out[-1])

        return {"val_mae_loss": self.loss_tracker_val.result(),
            "val_mae_h4pt": self.metric_h4pt_val.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.metric_h4pt]

def metric_dist(y_true, y_pred):
    """compute the l2 distance between corner
    Args:
        y_true (_type_): _description_
        y_pred (_type_): _description_
    """
    d = tf.reshape(y_true,(-1,4,2))-tf.reshape(y_pred,(-1,4,2))
    return tf.reduce_mean(tf.norm(d,ord='euclidean',axis=-1))