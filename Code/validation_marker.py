import numpy as np
import pandas as pd
import cv2

import torch
import torch.nn as nn
import torch.functional as F
import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model import *
from process import *
from Grad_CAM_module import *


def main(model, img, X_transform, target_category, i):
    
    model = model
    target_layers = [model.conv1.conv.conv]
    
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
   
    grayscale_cam = cam(X_transform, target_category)

    grayscale_cam = grayscale_cam[0, :]
    #print(grayscale_cam)
    mask_unit = np.uint8(grayscale_cam*255)
    max_val = np.max(mask_unit)

    threshold_value = int(max_val * 0.4)
    _, binary_mask = cv2.threshold(mask_unit, threshold_value, 255, cv2.THRESH_BINARY)

    plt.imshow(binary_mask,cmap='gray')
    plt.axis('off')
    plt.savefig('../Results/Grad-CAM/mask/segment_{}.png'.format(i), bbox_inches="tight", pad_inches=0)


    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig('../Results/Grad-CAM/grad/segment_{}.png'.format(i), bbox_inches="tight", pad_inches=0)



# load data
path_all = pd.read_csv('../images/SYSUCC/img_path.csv')
X = np.array(path_all['img_path'])

print(len(X))


# Image matrix
X_matrix = read_data(X, img_type='RGB',reshape_image=False)

# Images tensor
X_tensor = torch.tensor(np.array(X_matrix).transpose(0,3,1,2))

# load model
model = Combine_Model_SMLP(12966, 2)

model.load_state_dict(torch.load('../Results/model/net_save.pth'))


# Grad-CAM
for i in range(X_matrix.shape[0]):


    x_all = [X_tensor[i,:,:,:].unsqueeze(dim=0).float()] 
    
    # caf low related to TNBC
    # extract the caf low group marker for TNBC
    main(model, turn_images(X_tensor[i,:,:,:].unsqueeze(dim=0)), 
         x_all, int(0),i)
    
