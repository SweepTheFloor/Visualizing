# ----------COMPETITION LABELING------------
# Misc/Void                               0
# Road                                    1
# Sidewalk/Parking-slot/Terrain           2
# Fence/Construction-Objects-butNoBarrels 3
# Four-Wheel-Vehicle                      4
# TrafficSign                             5
# People                                  6
# Poles                                   7
# Barrels                                 8
# PotHoles (White Circles on floor)       9

import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
from draw_lane import combo_thresh, change_perspective,lr_curvature,draw_on_road
from calibrate import undist
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import pdb
from threshold_helpers import *
import math
import pdb
import msvcrt as m
def wait():
    m.getch()
    

#-----------Directories-------------
ROOT_DIR = os.path.dirname(os.getcwd())
#ROOT_DIR = os.getcwd()
CITYSCAPE_DIR = os.path.join(ROOT_DIR, "data" , "CITYSCAPES")  #MODIFY FOR HPC without S CITISCAPE'S' !!!!!!!!
RGB_CITIES_DIR = os.path.join(CITYSCAPE_DIR, "CityScape", "leftImg8bit","train")
GT_CITIES_DIR = os.path.join(CITYSCAPE_DIR, "CityScape", "gtLabels","train")

#-----------Get City names---------
cities = [fn for fn in os.listdir(GT_CITIES_DIR)]

for c_idx in range(len(cities)):#city index

    #-------Get images RGB, GT---------
    RGB_IMAGE_DIR = os.path.join(RGB_CITIES_DIR, cities[c_idx])
    RGB_images = [fn for fn in os.listdir(RGB_IMAGE_DIR) if fn.endswith('.png')]
    
    GT_IMAGE_DIR = os.path.join(GT_CITIES_DIR, cities[c_idx])
    GT_images = [fn for fn in os.listdir(GT_IMAGE_DIR) if fn.endswith('.txt')]
    
    for idx in range(len(GT_images)):#image index
    
        #------------LOAD RGB, GT, GTC IMAGES-----------------
        RGB_image_path = os.path.join(RGB_IMAGE_DIR , RGB_images[idx])
        RGB_img = cv2.imread(RGB_image_path)
        RGB_height, RGB_width, channels = RGB_img.shape
        print(RGB_img.shape)    
        
        GT_image_path = os.path.join(GT_IMAGE_DIR , GT_images[idx])
        GT_img = np.loadtxt(GT_image_path)    
        GT_height, GT_width = GT_img.shape
        print(GT_img.shape)
                
        
        ##------------ ROAD EXTRACTION----------
        road_mask = GT_img
        road_mask[road_mask != 1] = 0           #1 is the road values according to table
        road_mask = road_mask[:,:,np.newaxis]   #increase dimension to multiply mask later
    
        RGB_road = RGB_img * road_mask
        RGB_road = RGB_road.astype(RGB_img.dtype,copy=False) # making sure it remains depth == CV_8U || depth == CV_16U || depth == CV_32F

        ##-----------IDENTIFY LANES----------------
        '''
        load undistortion matrix from camera 
        '''
        with open('test_dist_pickle.p', 'rb') as pick:
          dist_pickle = pickle.load(pick)

        #mtx = dist_pickle['mtx']
        dist = dist_pickle['dist']
        fx = 2262.52
        fy = 2265.3017905988554
        cx = 1096.98
        cy = 513.137
        mtx = np.array([[fx,0,cx],
                        [0,fy,cy],
                        [0, 0, 1]])
                        
        undist_img = undist(RGB_img, mtx, dist)                          #apply threshold to original
        combo_image = combo_thresh(undist_img)
        
        undist_img = undist(RGB_road, mtx, dist)                         #apply threshold to road extraction
        combo_road_image = combo_thresh(undist_img)
        
        combo_road_image = cv2.bitwise_and(combo_road_image,combo_image) #remove contour from road extraction
        
        warped_image = change_perspective(combo_road_image)              #warp image
        
        out_img, left_fitx, lefty, right_fitx, righty, ploty, full_text = lr_curvature(warped_image)
        color_warp, result = draw_on_road(RGB_road, warped_image, left_fitx, lefty, right_fitx, righty, ploty)
        
        cv2.putText(result, full_text, (200, 100), cv2.FONT_HERSHEY_COMPLEX, 1, 255)    #add text to image
        
        
        #-----------Resize and show-------------
        Resized_RGB    = cv2.resize(RGB_img,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)    
        Resized_combo  = cv2.resize(combo_road_image,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)
        Resized_out    = cv2.resize(out_img,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)
        Resized_result = cv2.resize(result,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)
        
        Resized_height, Resized_width = Resized_combo.shape
        print(Resized_height, Resized_width)
        SideBySide = np.zeros((2*Resized_height , 2*Resized_width , 3), np.uint8)       #specifying uint8 is important
        
        Resized_combo = np.dstack((Resized_combo, Resized_combo, Resized_combo))*255  #make binary images-> rgb
        #Resized_warped = np.dstack((Resized_warped, Resized_warped, Resized_warped))*255  #make binary images-> rgb
        
        
        SideBySide[0:Resized_height,0:Resized_width]                = Resized_RGB[:,:]
        SideBySide[0:Resized_height,Resized_width:2*Resized_width]  = Resized_combo[:,:]
        SideBySide[Resized_height:2*Resized_height,0:Resized_width] = Resized_out[:,:]
        SideBySide[Resized_height:2*Resized_height,Resized_width:2*Resized_width] = Resized_result[:,:]
        
        # SideBySide = np.array(SideBySide,np.uint8)
        # # RGB_img_lanes = np.array(RGB_img_lanes)
        # # im = Image.fromarray(RGB_img_lanes)
        # # im.show()
        # # wait()
        # # im.close()
        
        # #vvv
        # # cv2.startWindowThread()
        # # cv2.namedWindow('Virtual World')
        cv2.imshow('Virtual World',SideBySide)
        key = cv2.waitKey(0)
        if key==27: 
            cv2.destroyAllWindows()
            quit()
        
        