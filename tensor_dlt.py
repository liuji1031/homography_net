import tensorflow as tf
import numpy as np
import cv2

def TensorDLT(h4pt_batch : tf.Tensor,batch_size=8, crop_size=128, debug=False):
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
    
    def expand_dim(m):
        m = tf.expand_dims(m, axis=0)
        m = tf.repeat(m, repeats=batch_size, axis=0)
        return m

    uv_prime = expand_dim(uv_prime)
    
    duv = tf.reshape(h4pt_batch, (-1,4,2))
    uv = uv_prime - duv # 4 corners mapped from 4 points of the rectangular 
                        # image patch by applying H
    
    # flip both uv and uv_prime, such that each row is xy instead of row, col
    uv = tf.reverse(uv, axis=[-1])
    uv_prime = tf.reverse(uv_prime, axis=[-1])
    
    # in other words, uv_prime = H * uv

    for i in range(4):
        uv1 = tf.concat((uv[:,i,:][:,tf.newaxis,:], tf.ones((batch_size, 1,1))),
                        axis=-1) 
        # size is batchsize by 1 by 3

        tmp1 = tf.concat((tf.zeros((batch_size,1,3)),-1.0*uv1),axis=-1)
        tmp2 = tf.concat((uv1,tf.zeros((batch_size,1,3))),axis=-1)
        tmp = tf.concat((tmp1,tmp2),axis=-2)
        assert(tmp.shape==(batch_size,2,6))

        tmp3 = tf.convert_to_tensor(
            np.array([1.0,-1.0])[np.newaxis,:,np.newaxis],
                                    dtype=tf.float32)
        
        # to get tmp3 equals [vprime,-uprime]^T
        tmp3 = tmp3 * tf.reverse(uv_prime[:,i,:],axis=[-1])[:,:,tf.newaxis]
        assert(tmp3.shape==(batch_size,2,1))
        tmp3_ = uv[:,i,:][:,tf.newaxis,:]
        assert(tmp3_.shape==(batch_size,1,2))
        tmp4 = tf.matmul(tmp3,tmp3_)
        
        tmp = tf.concat((tmp,tmp4),axis=-1)
        assert(tmp.shape==(batch_size,2,8))

        if A is None:
            A = tmp
        else:
            A = tf.concat((A, tmp),axis=-2)
        
        if b is None:
            b = -1.0*tmp3 # b equals [-vprime, uprime]^T
        else:
            b = tf.concat((b, -1.0*tmp3),axis=-2)

    # solve for H using pseudo inv
    H = tf.matmul(tf.linalg.pinv(A), b) # batch size by 8 by 1
    H = tf.reshape(tf.concat((H,tf.ones((batch_size,1,1))),axis=-2),(-1,3,3))

    if debug:
        for b in range(batch_size):
            uv_ = tf.squeeze(uv[b,:,:]).numpy()
            uv_p_ = tf.squeeze(uv_prime[b,:,:]).numpy()
            H_ = cv2.getPerspectiveTransform(uv_,
                                             uv_p_)
            print("expected:\n",H_)

            Hc = tf.squeeze(H[b,:,:]).numpy()
            print("calculated:\n",H[b,:].numpy())

            d = np.linalg.norm(H_-Hc) / np.linalg.norm(H_)
            print(f"relative diff: {d:.2e}")
            for i in range(4):
                tmp = Hc.dot(np.array([uv_[i,0],
                                        uv_[i,1],
                                        1.0])[:,np.newaxis])
                print(uv_p_[i,:],np.round(tmp[:2].flatten()/tmp[2]))
    
    return H
