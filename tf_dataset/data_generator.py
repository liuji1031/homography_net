from random import shuffle
from typing import Any
import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self,
                 path,
                 crop_size=128,
                 rho=32,
                 resize_size=(320,240),
                 mode='supervised',
                 do_resize=True):
        self.crop_size = crop_size
        self.resize_shape = resize_size
        self.rho = rho
        self.mode = mode
        self.do_resize = do_resize

        # find all images in the path
        if isinstance(path,str):
            path = Path(path)
        self.im_list = list(path.glob("*.jpg"))
        self.nimg = len(self.im_list)
        self.ifile=0

    def pass_colinear(self, pts):
        a,b,c = pts[0],pts[1],pts[2]
        d = np.flip(c-a)
        d = d / np.linalg.norm(d)
        d[1] = -d[1]
        proj_bd = np.inner(b,d)
        proj_ad = np.inner(a,d)
        proj_cd = np.inner(c,d)

        if proj_bd > proj_ad+5 and proj_bd > proj_cd+5:
            return True
        else:
            print(pts)
            print(proj_bd, proj_ad, proj_cd)
            return False

    def gen_img_and_homography(self,img_path,debug=False):
    
        # read image
        im = Image.open(img_path)
        if self.do_resize:
            im = im.resize(self.resize_shape) # resize
        im_ori = np.array(im)

        if debug:
            plt.imshow(im_ori/255.0)
            plt.show()

        if im_ori.ndim < 3: # skip single channel
            return None, None
        h,w = im_ori.shape[:2]

        # randomly select the corners for cropping
        # making sure the shifted corners are within the image
        if isinstance(self.crop_size,int):
            ch = self.crop_size
            cw = self.crop_size
        else: # list of crop size
            ch,cw = self.crop_size

        if h-self.rho-ch <= self.rho or w-self.rho-cw <= self.rho:
            return None, None

        # choose upper left corner
        upper_left_h = np.random.randint(low=self.rho,high=h-self.rho-ch)
        upper_left_w = np.random.randint(low=self.rho,high=w-self.rho-cw)
        
        upper_left_coord = tf.convert_to_tensor(np.array([upper_left_h,
                                                          upper_left_w]))

        corner_pts = np.array([[0,0],[ch-1,0],[ch-1,cw-1],[0,cw-1]]) + \
                    np.array([upper_left_h, upper_left_w])[np.newaxis,:]
        
        # generate the new 4 corner points
        regen = True
        while regen:
            corner_pts_new = np.copy(corner_pts)
            corner_pts_new += np.random.randint(-self.rho,
                                                    self.rho+1,
                                                    size=(4,2))
            ind = np.array([0,1,2,3])
            for k in range(4):
                pts = np.copy(corner_pts_new)
                pts = pts[(ind+k)%4,:]
                pts = pts[1:,:]-pts[[0],:]
                if not self.pass_colinear(pts):
                    print('colinearity detected!')
                    break
            regen=False

        # note H map from corners new back to original corners

        # calculate the difference in xy coordinate
        H4pt = (corner_pts - corner_pts_new).flatten()
        if debug is True:
            print(corner_pts - corner_pts[[0],:])
            print(corner_pts_new -corner_pts[[0],:])
            print(H4pt)

        # get actual homography
        H = cv2.getPerspectiveTransform(
            src=np.fliplr(corner_pts_new.astype(np.float32)),
            dst=np.fliplr(corner_pts.astype(np.float32)))
        
        im_warp = cv2.warpPerspective(im_ori, H, (w,h))

        # get the two patches
        p1 = im_ori[upper_left_h:upper_left_h+ch,upper_left_w:upper_left_w+cw]
        p2 = im_warp[upper_left_h:upper_left_h+ch,upper_left_w:upper_left_w+cw]

        # convert to tensor
        p1 = tf.convert_to_tensor(p1, dtype=tf.float32) #/ 255.0
        p2 = tf.convert_to_tensor(p2, dtype=tf.float32) #/ 255.0
        im_ori = tf.convert_to_tensor(im_ori/255., dtype=tf.float32)
        im_warp = tf.convert_to_tensor(im_warp/255., dtype=tf.float32)

        if debug:
            print(H)
            # sanity check, check the coordinate of the corners
            for i in range(4):
                tmp = H.dot(np.array([corner_pts_new[i,1],
                                            corner_pts_new[i,0],
                                            1.0])[:,np.newaxis])
                print(corner_pts[i,:],np.flipud(tmp[:2]).flatten()/tmp[2])
            plt.imshow(np.hstack((p1/255, p2/255)))
        
        h4pt = tf.convert_to_tensor(H4pt,dtype=tf.float32)
        if self.mode == "unsupervised_test":
            input = (p1,p2,im_ori,upper_left_coord,h4pt)
            output = im_warp
        elif self.mode == "unsupervised_with_h4pt":
            # under this mode, output the 4 point representation
            # as well, so we can calculate metric 
            input = (p1,p2,im_ori,upper_left_coord)
            output = (im_warp, h4pt)
        elif self.mode == "unsupervised":
            input = (p1,p2,im_ori,upper_left_coord)
            output = im_warp
        elif self.mode == "supervised":
            input = (p1,p2)
            output = h4pt

        return input, output

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        while True:
            if self.ifile==0:
                # reshuffle every time the data are exhausted
                # print("reshuffling dataset")
                shuffle(self.im_list)
            im_path = self.im_list[self.ifile]
            self.ifile = (self.ifile+1)%self.nimg
            input_data, output_data = self.gen_img_and_homography(im_path,
                                                                  **kwds)
            if input_data is None:
                continue
            yield input_data, output_data