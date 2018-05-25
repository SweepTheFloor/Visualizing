import cv2
import pickle
import numpy as np
import scipy.misc as sci
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
#imageio.plugins.ffmpeg.download()
#from moviepy.editor import VideoFileClip
#from moviepy.editor import *
import pdb
#import helper methods from other files 
from calibrate import undist
from threshold_helpers import *

a = np.array([1,0,3])
print(a.nonzero())

# with open('test_dist_pickle.p', 'rb') as pick:
  # dist_pickle = pickle.load(pick)

# #mtx = dist_pickle['mtx']
# dist = dist_pickle['dist']

# fx = 2262.52
# fy = 2265.3017905988554
# cx = 1096.98
# cy = 513.137
# mtx = np.array([[fx,0,cx],
                # [0,fy,cy],
                # [0, 0, 1]])
                

# img = mpimg.imread('testImageLaneCalibration.png')

# undist_img = undist(img, mtx, dist)
# # plt.imshow(undist_img)
# # plt.title('undist_img')
# # plt.show()

# height, width = undist_img.shape[0], undist_img.shape[1]
# print height, width
# roi = undist_img[np.int32(4.0/7.0*height):np.int32(6.0/7.0*height), 0:width]
# # plt.imshow(roi)
# # plt.title('roi')
# # plt.show()

# IMAGE_H = roi.shape[0]
# IMAGE_W = roi.shape[1]

# src = np.float32([[0, IMAGE_H], [2008, IMAGE_H], [700, 0], [IMAGE_W-620, 0]])
# dst = np.float32([[0, IMAGE_H], [IMAGE_W, IMAGE_H], [0, 0], [IMAGE_W, 0]])
# M = cv2.getPerspectiveTransform(src, dst) # The transformation matrix
# Minv = cv2.getPerspectiveTransform(dst, src) # Inverse transformation


# warped_img = cv2.warpPerspective(roi, M, (IMAGE_W, IMAGE_H)) # Image warping
# plt.imshow(cv2.cvtColor(warped_img, cv2.COLOR_BGR2RGB)) # Show results
# plt.show()

