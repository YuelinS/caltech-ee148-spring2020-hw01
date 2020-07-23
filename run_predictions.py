assert False

import os
import numpy as np
import json
from PIL import Image, ImageDraw 
from matplotlib import pyplot as plt

assert False

# Red light filter parameter tuning:
    
# filter shape params: red_diameter, black_edge, blue_padding
filter_paras = [(7,2,0),(7,2,3),(23,3,0),(23,3,5)]
strides = [5,5,10,10]
thres = [-1000,5000,0,10000]
   
# filter color params:
k = 140
sky = [150-k,180-k,200-k]
# sky = [100, 200, 255]

filter_w_red = 300
filter_w_blk_upper, filter_w_blk_lower = -100, -50

# using filter #:
flag_filter = range(4) # [0]  #
flag_figs = [1] #range(334)

def detect_red_light(Iasarray,filters,strides):
    '''
    This function takes a numpy array <I> and returns a list <bounding_boxes>.
    The list <bounding_boxes> should have one element for each red light in the 
    image. Each element of <bounding_boxes> should itself be a list, containing 
    four integers that specify a bounding box: the row and column index of the 
    top left corner and the row and column index of the bottom right corner (in
    that order). See the code below for an example.
    
    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''
        
    bounding_boxes = [] # This should be a list of lists, each of length 4. See format example below. 
    bounding_box_filter_idx = []
    top_convs = np.empty((10,4))
    
    (im_rows,im_cols,im_channels) = np.shape(Iasarray)

    for i in flag_filter:      
        # print(f'filter_{i}')
        
        filter_rgb = filters[i]
        (filter_rows,filter_cols,filter_channels) = np.shape(filter_rgb)
    
        stride = strides[i]
        thre = thres[i]
        
        convs = []
        rows = []
        cols = []
        
        for row in range(0,im_rows - filter_rows - drop_bottom,stride):
            
            for col in range(0,im_cols - filter_cols,stride):
        
                    patch = Iasarray[row : row + filter_rows  , col : col + filter_cols]
        
                    # normalize
                    # norm = np.linalg.norm(patch)    
                    # patch = patch / norm
    
                    norm = max(np.amax(patch) - np.amin(patch), 200)
                    patch = (patch)/ norm  # - np.amin(patch)
                    
                    convs.append(np.sum(patch*filter_rgb))
                    rows.append(row)
                    cols.append(col)
        
        # decide threshold
        conv_row = (im_rows - filter_rows - drop_bottom)//stride + int((im_rows - filter_rows - drop_bottom) % stride != 0)
        conv_col = (im_cols - filter_cols)//stride + + int((im_cols - filter_cols) % stride != 0)
        conv_img = np.array(convs).reshape(conv_row,conv_col)
        plt.imshow(conv_img)
        plt.colorbar()
        rgb_range = conv_img.copy().flatten()
        rgb_range.sort()
        plt.title(np.histogram(rgb_range, bins=5)[1], size=10)
        plt.xlabel(rgb_range[-10:][::-1])
        plt.show()
        
        sort_convs = np.sort(convs)[::-1]
        sort_convs_idx = np.argsort(convs)[::-1]
        top_convs[:,i] = sort_convs[:10]
        
        select_convs = sort_convs > thre       
        # assert sum(select_convs) > 10
        
        pick_rows = np.array(rows)[sort_convs_idx[select_convs]]
        pick_cols = np.array(cols)[sort_convs_idx[select_convs]]
       
    
        for j in range(len(pick_rows)):
            
            tl_row = int(pick_rows[j])
            tl_col = int(pick_cols[j])
            br_row = tl_row + filter_rows
            br_col = tl_col + filter_cols
            bounding_box_filter_idx.append(i)
            bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes, top_convs, bounding_box_filter_idx



def make_red_light_filter(filter_paras):
    
    filters = []

    for i in range(len(filter_paras)):
    
        # make the red light filter
        dia = filter_paras[i][0]
        edge = filter_paras[i][1]
        pad = filter_paras[i][2]
        
        rad = int(np.floor(dia/2))
        
        filter_rows = dia*3+edge
        filter_cols = dia+edge*2
        
        circle_center = np.array((rad+edge,rad+edge))
        
        filter_rgb_upper = np.ones((dia,filter_cols,3)) * filter_w_blk_upper
        filter_rgb_lower = np.ones((dia*2+edge,filter_cols,3)) * filter_w_blk_lower
        filter_rgb = np.vstack([filter_rgb_upper, filter_rgb_lower])

        filter_circle_mask = np.array([[(np.sum((np.array((row,col))-circle_center)**2) <= rad**2) for col in range(filter_cols)] for row in range(filter_rows)])     
        
        # filter_rgb = np.ones((dia+edge*2,dia+edge*2,3))*(-10)
        # filter_circle_mask = np.array([[(np.sum((np.array((row,col))-circle_center)**2) <= rad**2) for col in range(dia)] for row in range(dia)])     
                     
        
        filter_rgb[filter_circle_mask,0]= filter_w_red
        
        filter_pad = np.empty((filter_rows+pad*2,filter_cols+pad*2,3))
        
        for channel in range(3):
        
            filter_pad[:,:,channel] = np.pad(filter_rgb[:,:,channel], (pad,pad), 'constant', constant_values=[(sky[channel],sky[channel])])
        
        
        # smooth
        
        # make red light filter by image
        # I = Image.open(os.path.join(data_path,file_names[1]))
        # I.show()
            
        plt.imshow(filter_pad/255)
        # plt.show()
        plt.savefig(os.path.join(preds_path,'red_light_filter_' + f'{i}' + '.png'))
    
        filters.append(filter_pad)
    
    
    
    return filters
    
       
    
    
#%%

# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

results_path = '../results/hw01'
os.makedirs(results_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 



# make filter
filters = make_red_light_filter(filter_paras)



drop_bottom = 72
    
preds = {}

for i in flag_figs:   #len(file_names)):  #
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    Iasarray = np.asarray(I)
    
    # main function
    bounding_boxes, top_convs, bounding_box_filter_idx = detect_red_light(Iasarray,filters,strides)
    
    preds[file_names[i]] = bounding_boxes
    
    # visualization
    img = ImageDraw.Draw(I)  #Image.fromarray(I))  

    for bounding_box, j in zip(bounding_boxes, bounding_box_filter_idx):
        bounding_box = [bounding_box[k] for k in [1,0,3,2]]
        img.rectangle(bounding_box, outline ="hsl({}, 100%, 50%)".format(j*90))
        
    # I.show()
    I.save(os.path.join(results_path,file_names[i]))
    
    

# save preds (overwrites any previous predictions!)
with open(os.path.join(results_path,'preds.json'),'w') as f:
    json.dump(preds,f, indent=4)



