"""
======================================================================
code credit: https://github.com/kevinzakka/spatial-transformer-network

======================================================================

the grid generation is modified for use with homography. the original
code only deals with affine transformation.

"""

import sys
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from loguru import logger

def spatial_transformer_network(input_fmap,
                                homography,
                                img_width=128,
                                img_height=128,
                                out_dims=None,
                                **kwargs):
    """
    Spatial Transformer Network layer implementation as described in [1].

    The layer is composed of 3 elements:

    - localization_net: takes the original image as input and outputs
      the parameters of the affine transformation that should be applied
      to the input image.

    - grid_generator: generates a grid of (x,y) coordinates that
      correspond to a set of points where the input should be sampled
      to produce the transformed output.

    - bilinear_sampler: takes as input the original image and the grid
      and produces the output image using bilinear interpolation.

    Input
    -----
    - input_fmap: output of the previous layer. Can be input if spatial
      transformer layer is at the beginning of architecture. Should be
      a tensor of shape (B, H, W, C).

    - homography: homography transform tensor of shape (B, 9)

    Returns
    -------
    - out_fmap: transformed input feature map. Tensor of size (B, H, W, C).

    Notes
    -----
    [1]: 'Spatial Transformer Networks', Jaderberg et. al,
         (https://arxiv.org/abs/1506.02025)

    """
    # grab input dimensions
    B = tf.shape(input_fmap)[0]
    H = img_height
    W = img_width

    # reshape homography to (B, 2, 3)
    homography = tf.reshape(homography, [B, 3, 3])

    # generate grids of same size or upsample/downsample if specified
    if out_dims:
        out_H = out_dims[0]
        out_W = out_dims[1]
        batch_grids = grid_generator_v2(out_H, out_W, homography)
    else:
        batch_grids = grid_generator_v2(H, W, homography)

    x_s = batch_grids[:, 0, :, :]
    y_s = batch_grids[:, 1, :, :]

    # sample input with grid to get output
    out_fmap = bilinear_sampler(input_fmap, x_s, y_s)

    return out_fmap

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.

    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)

    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def grid_generator_v2(height, width, homography):
    num_batch = tf.shape(homography)[0]

    # create normalized 2D grid
    x = tf.linspace(-1.0, 1.0, width)
    y = tf.linspace(-1.0, 1.0, height)
    x_t, y_t = tf.meshgrid(x, y)

    # flatten
    x_t_flat = tf.reshape(x_t, [-1])
    y_t_flat = tf.reshape(y_t, [-1])

    # reshape to [x_t, y_t , 1] - (homogeneous form)
    ones = tf.ones_like(x_t_flat)
    sampling_grid = tf.stack([x_t_flat, y_t_flat, ones])

    # repeat grid num_batch times
    sampling_grid = tf.expand_dims(sampling_grid, axis=0)
    sampling_grid = tf.tile(sampling_grid, tf.stack([num_batch, 1, 1]))

    # cast to float32 (required for matmul)
    homography = tf.cast(homography, 'float32')
    sampling_grid = tf.cast(sampling_grid, 'float32')

    # transform the sampling grid - batch multiply
    w = float(width)
    h = float(height)
    M = tf.constant([[w/2.,0.,w/2.],
                     [0.,h/2,h/2.],
                     [0.,0.,1.]],dtype=tf.float32)
    Minv = tf.linalg.pinv(M,name='inv3')

    def expand_mat(mat):
        mat = tf.expand_dims(mat,axis=0)
        mat = tf.repeat(mat,num_batch,axis=0)
        return mat
    
    M = expand_mat(M)
    Minv = expand_mat(Minv)

    Hinv = tf.linalg.matmul(Minv, tf.linalg.inv(homography))
    Hinv = tf.linalg.matmul(Hinv, M)

    batch_grids = tf.matmul(Hinv, sampling_grid)
    # batch grid has shape (num_batch, 3, H*W)

    # normalize by scale factor
    batch_grids = batch_grids[:,:2,:] / batch_grids[:,2,:][:,tf.newaxis,:]
    # batch grid has shape (num_batch, 2, H*W)

    batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

    return batch_grids

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.

    To test if the function works properly, output image should be
    identical to input image when homography is initialized to identity
    transform.

    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of grid_generator.

    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    out = tf.where(tf.math.is_nan(out), 0., out)
    print("checking nan")
    tf.debugging.check_numerics(out,message='')

    return out