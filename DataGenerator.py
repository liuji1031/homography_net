from random import shuffle
from typing import Any
import tensorflow as tf
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class DataGenerator:
    def __init__(self, path, crop_size=128, rho=16, mode='supervised'):
        self.crop_size = crop_size
        self.rho = rho
        self.mode = mode

        # find all images in the path
        if isinstance(path,str):
            path = Path(path)
        self.im_list = list(path.glob("*.jpg"))
        self.nimg = len(self.im_list)
        shuffle(self.im_list)
        self.ifile=0

    def gen_img_and_homography(self,img_path,debug=False):
    
        # read image
        im_data = np.array(Image.open(img_path))
        if im_data.ndim < 3: # skip single channel
            return None, None
        h,w = im_data.shape[:2]

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
        
        # Coordinates of Patch A
        C_A_4pts = np.array([[upper_left_h, upper_left_w],              # upper left
                             [upper_left_h+ch-1, upper_left_w],         # bottom left
                             [upper_left_h+ch-1, upper_left_w+cw-1],    # bottom right
                             [upper_left_h, upper_left_w+cw-1]])        # upper right
                             

        corner_pts = np.array([[0,0],[ch-1,0],[ch-1,cw-1],[0,cw-1]]) + \
                    np.array([upper_left_h, upper_left_w])[np.newaxis,:]
        corner_pts_new = np.copy(corner_pts)
        # generate the new 4 corner points
        for i in range(4):
            corner_pts_new[i,:] += np.random.randint(-self.rho,
                                                     self.rho+1,
                                                     size=(2,))
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
        
        im_warp = cv2.warpPerspective(im_data, H, (w,h))

        # get the two patches
        p1 = im_data[upper_left_h:upper_left_h+ch,upper_left_w:upper_left_w+cw]
        p2 = im_warp[upper_left_h:upper_left_h+ch,upper_left_w:upper_left_w+cw] 

        p1 = tf.convert_to_tensor(p1, dtype=tf.float32)
        p2 = tf.convert_to_tensor(p2, dtype=tf.float32)

        if debug:
            print(H)
            # sanity check, check the coordinate of the corners
            for i in range(4):
                tmp = H.dot(np.array([corner_pts_new[i,1],
                                            corner_pts_new[i,0],
                                            1.0])[:,np.newaxis])
                print(corner_pts[i,:],np.flipud(tmp[:2]).flatten()/tmp[2])
            plt.imshow(np.hstack((p1/255, p2/255)))

        if self.mode == "unsupervised":
            input = (p1,p2)
            output = (p2, C_A_4pts)
        elif self.mode == "supervised":
            input = (p1,p2)
            output = tf.convert_to_tensor(H4pt,dtype=tf.float32)

        return input, output

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        while True:
            im_path = self.im_list[self.ifile]
            self.ifile = (self.ifile+1)%self.nimg
            im_crop, h4pt = self.gen_img_and_homography(im_path, **kwds)
            if im_crop is None:
                continue
            yield im_crop, h4pt