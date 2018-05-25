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


import msvcrt as m
def wait():
    m.getch()
    

#Directories and RGB & GT images
ROOT_DIR = os.path.dirname(os.getcwd())
#ROOT_DIR = os.getcwd()
CITYSCAPE_DIR = os.path.join(ROOT_DIR, "data" , "CITYSCAPES")  #MODIFY FOR HPC without S CITISCAPE'S' !!!!!!!!

RGB_CITIES_DIR = os.path.join(CITYSCAPE_DIR, "CityScape", "leftImg8bit","train")
GT_CITIES_DIR = os.path.join(CITYSCAPE_DIR, "CityScape", "gtLabels","train")
GTC_CITIES_DIR = os.path.join(CITYSCAPE_DIR, "CityScape", "gtFine","train")

#-----------Get City names---------
cities = [fn for fn in os.listdir(GT_CITIES_DIR)]

for c_idx in range(len(cities)):#city index

    #-------Get images RGB, GT, GTC---------
    RGB_IMAGE_DIR = os.path.join(RGB_CITIES_DIR, cities[c_idx])
    RGB_images = [fn for fn in os.listdir(RGB_IMAGE_DIR) if fn.endswith('.png')]
    
    GT_IMAGE_DIR = os.path.join(GT_CITIES_DIR, cities[c_idx])
    GT_images = [fn for fn in os.listdir(GT_IMAGE_DIR) if fn.endswith('.txt')]
    
    GTC_IMAGE_DIR = os.path.join(GTC_CITIES_DIR, cities[c_idx])
    GTC_images = [fn for fn in os.listdir(GTC_IMAGE_DIR) if fn.endswith('color.png')]

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
        
        GTC_image_path = os.path.join(GTC_IMAGE_DIR , GTC_images[idx])
        GTC_img = cv2.imread(GTC_image_path)    
        GTC_height, GTC_width, channels = GTC_img.shape
        print(GTC_img.shape)
        
        
        ##------------Get Only ROAD ----------
        road_mask = GT_img
        road_mask[road_mask != 1] = 0           #1 is the road values according to table
        road_mask = road_mask[:,:,np.newaxis]   #increase dimension to multiply mask later
    
        RGB_road = RGB_img * road_mask
        GTC_road = GTC_img * road_mask

            
        ##-----------Resize and show-------------
        Resized_RGB = cv2.resize(RGB_img,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)    
        Resized_GTC = cv2.resize(GTC_img,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)    
        
        Resized_RGB_road = cv2.resize(RGB_road,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)    
        Resized_GTC_road = cv2.resize(GTC_road,(np.int(.3*RGB_width),np.int(.3*RGB_height)), interpolation = cv2.INTER_CUBIC)    
        
        
        Resized_GTC_height, Resized_GTC_width, channels = Resized_GTC.shape
        
        SideBySide = np.zeros((2*Resized_GTC_height , 2*Resized_GTC_width , 3), np.uint8)       #specifying uint8 is important
        SideBySide[0:Resized_GTC_height,0:Resized_GTC_width] = Resized_RGB[:,:]
        SideBySide[0:Resized_GTC_height,Resized_GTC_width:2*Resized_GTC_width] = Resized_GTC[:,:]
        SideBySide[Resized_GTC_height:2*Resized_GTC_height,0:Resized_GTC_width] = Resized_RGB_road[:,:]
        SideBySide[Resized_GTC_height:2*Resized_GTC_height,Resized_GTC_width:2*Resized_GTC_width] = Resized_GTC_road[:,:]
        
        SideBySide = np.array(SideBySide,np.uint8)
        im = Image.fromarray(SideBySide)
        im.show()
        wait()
        
        #vvv
        # cv2.startWindowThread()
        # cv2.namedWindow('Virtual World')
        # cv2.imshow('Virtual World',SideBySide)
        # key = cv2.waitKey(0)
        # if key==27: 
            # cv2.destroyAllWindows()
            # quit()
        
        