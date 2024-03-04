import tensorflow as tf
from pathlib import Path
from tf_dataset.data_generator import DataGenerator

def config_ds(ds,batch_size=8):
    AUTOTUNE = tf.data.AUTOTUNE
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    ds = ds.batch(batch_size)
    return ds

def get_tf_dataset(path,
                   batch_size=8,
                   im_crop_shape=(128,128,3),
                   im_ori_shape=(240,320,3),
                   mode="supervised",
                   do_resize=True,
                   rho=32):
    mapping = {"supervised":"supervised",
               "unsupervised":"unsupervised_with_h4pt"}
    data_generator = DataGenerator(path,mode=mapping[mode],do_resize=do_resize,
                                   rho=rho)

    if mode=="supervised":
        output_signature = (
                    # input
                    (tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32),
                     tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32)
                    ),
                    # output
                     tf.TensorSpec(shape = (8,),dtype=tf.float32),
                    )
    elif mode=="unsupervised":
        output_signature=(  #input
                    (tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32),
                    tf.TensorSpec(shape=im_crop_shape,dtype=tf.float32),
                    tf.TensorSpec(shape=im_ori_shape,dtype=tf.float32),
                    tf.TensorSpec(shape=(2,),dtype=tf.float32),
                    ),
                     # output
                    (tf.TensorSpec(shape=im_ori_shape,dtype=tf.float32), 
                     tf.TensorSpec(shape=(8,),dtype=tf.float32))
                    )

    ds = tf.data.Dataset.from_generator(data_generator,
                                        output_signature=output_signature)

    ds = config_ds(ds, batch_size=batch_size)
    return ds