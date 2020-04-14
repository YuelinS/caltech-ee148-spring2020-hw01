#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 00:12:50 2020

@author: Lenovo
"""
import os
import numpy as np
import json
from PIL import Image, ImageDraw 
import numpy as np


#%%
# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'


# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 

# set a path for saved predictions: 
preds_path = './' 

results_path = '../results/hw01'
os.makedirs(results_path,exist_ok=True) # create directory if needed 

# read preds 
with open(os.path.join(preds_path,'preds.json'),'r') as f:
    preds = json.load(f)
    


# the red light filter
    
bounding_boxes = preds[file_names[1]]
bounding_box = bounding_boxes[0]


filter_rows = bounding_box[2] - bounding_box[0]
filter_cols = bounding_box[3] - bounding_box[1]


#%%    
for i in range(1,2):   #len(file_names)):  #
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    bounding_boxes = preds[file_names[i]]
    img = ImageDraw.Draw(I)  #Image.fromarray(I))  

    for j in range(len(bounding_boxes)):
        bounding_box = bounding_boxes[j]
        bounding_box = [ bounding_box[i] for i in [1,0,3,2]]  
        img.rectangle(bounding_box, outline ="red") 
        
    I.show()
    # preds[file_names[i]] = detect_red_light(I,filter_rgb)

    I.save(os.path.join(results_path,file_names[i]))
