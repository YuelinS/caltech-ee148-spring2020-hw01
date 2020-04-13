import os
import numpy as np
import json
from PIL import Image
from matplotlib import pyplot as plt


def detect_red_light(I,filter_rgb):
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
        
    
    (im_rows,im_cols,im_channels) = np.shape(I)
    
    convs = []
    rows = []
    cols = []
    
    for row in range(0,im_rows - filter_rows,stride):
        
        for col in range(0,im_cols - filter_cols,stride):
    
                patch = I[row : row + filter_rows  , col : col + filter_cols]
    
                convs.append(np.sum(patch*filter_rgb))
                rows.append(row)
                cols.append(col)
    
    # decide threshold
    
    plt.plot(convs)
    plt.show()
    
    sort_convs = np.argsort(convs)[::-1][0:7]
    pick_rows = np.array(rows)[sort_convs]
    pick_cols = np.array(cols)[sort_convs]
       
    
    for i in range(len(pick_rows)):
        
        
        tl_row = int(pick_rows[i])
        tl_col = int(pick_cols[i])
        br_row = tl_row + filter_rows
        br_col = tl_col + filter_cols
        
        bounding_boxes.append([tl_row,tl_col,br_row,br_col]) 
    
    
    for i in range(len(bounding_boxes)):
        assert len(bounding_boxes[i]) == 4
    
    return bounding_boxes



# set the path to the downloaded data: 
data_path = '../data/RedLights2011_Medium'

# set a path for saving predictions: 
preds_path = './' 
os.makedirs(preds_path,exist_ok=True) # create directory if needed 

# get sorted list of files: 
file_names = sorted(os.listdir(data_path)) 

# remove any non-JPEG files: 
file_names = [f for f in file_names if '.jpg' in f] 


# make the red light filter
dia = 41 
rad = int(np.floor(dia/2))
circle_center = np.array((rad,rad))

filter_rgb = np.zeros((dia*3,dia,3))
filter_circle_mask = np.array([[(np.sum((np.array((row,col))-circle_center)**2) <= rad**2) for col in range(dia)] for row in range(dia*3)])     
             

filter_rgb[filter_circle_mask,0]=250    
plt.imshow(filter_rgb)
    
(filter_rows,filter_cols,filter_channels) = np.shape(filter_rgb)

# convolution stride
stride = 10
    
preds = {}
for i in range(1):   #len(file_names)):  #
    
    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names[i]))
    
    # convert to numpy array:
    I = np.asarray(I)
    
    preds[file_names[i]] = detect_red_light(I,filter_rgb)

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds.json'),'w') as f:
    json.dump(preds,f)
