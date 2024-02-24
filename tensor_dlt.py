import tensorflow as tf
import numpy as np
import cv2

def TensorDLT(h4pt : tf.Tensor, crop_size=128, debug=False):
    """recover homography H from the 4 point representation and maintain
    differentiability
    """

    # construct the A matrix
    s = crop_size
    A,b = None,None

    # 4 corners of the image patch
    uv_prime = tf.convert_to_tensor(np.array([[0,0],
                                              [s-1,0],
                                              [s-1,s-1],
                                              [0,s-1]]),
                                              dtype=tf.float32)
    
    duv = tf.reshape(h4pt, (4,2))
    uv = uv_prime - duv # 4 corners mapped from 4 points of the rectangular 
                        # image patch by applying H
    
    # flip both uv and uv_prime, such that each row is xy instead of row, col
    uv = tf.reverse(uv, axis=[1])
    uv_prime = tf.reverse(uv_prime, axis=[1])
    
    # in other words, uv_prime = H * uv

    for i in range(4):
        uv1 = tf.concat((uv[i,:][tf.newaxis,:], tf.ones((1,1))),axis=1) # 1 by 3

        tmp1 = tf.concat((tf.zeros((1,3)),-1.0*uv1),axis=1)
        tmp2 = tf.concat((uv1,tf.zeros((1,3))),axis=1)
        tmp = tf.concat((tmp1,tmp2),axis=0)
        assert(tmp.shape==(2,6))

        tmp3 = tf.convert_to_tensor(np.array([1.0,-1.0])[:,np.newaxis],
                                    dtype=tf.float32)
        
        # tmp3 equals [vprime,-uprime]^T
        tmp3 = tmp3 * tf.reverse(uv_prime[i,:],axis=[0])[:,tf.newaxis]
        assert(tmp3.shape==(2,1))
        tmp4 = tf.matmul(tmp3,uv[i,:][tf.newaxis,:])
        
        tmp = tf.concat((tmp,tmp4),axis=1)
        assert(tmp.shape==(2,8))

        if A is None:
            A = tmp
        else:
            A = tf.concat((A, tmp),axis=0)
        
        if b is None:
            b = -1.0*tmp3 # b equals [-vprime, uprime]^T
        else:
            b = tf.concat((b, -1.0*tmp3),axis=0)

    # solve for H using pseudo inv
    H = tf.matmul(tf.linalg.pinv(A), b)
    H = tf.reshape(tf.concat((H,tf.ones((1,1))),axis=0),(3,3))

    if debug:
        H_ = cv2.getPerspectiveTransform(uv.numpy(),
                                            uv_prime.numpy())
        print("expected:\n",H_)
        print("calculated:\n",H)
        for i in range(4):
            tmp = H.numpy().dot(np.array([uv.numpy()[i,0],
                                    uv.numpy()[i,1],
                                    1.0])[:,np.newaxis])
            print(uv_prime[i,:].numpy(),np.round(tmp[:2].flatten()/tmp[2]))
    
    return H
