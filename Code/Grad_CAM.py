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

    
# Load data
# gene data
data_gene = pd.read_csv("../Data/scATOMIC_Macro/breast_expr_bayes8.csv",index_col=0)

# data normalization
data_gene = data_gene.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

# images path
path_all = '../images/TCGA/BRCA_select/'

X = []
Y = []

for patient_i in os.listdir(path_all):
        
    for image_i in os.listdir(path_all+patient_i):
        Y.append(patient_i)
        X.append(path_all+ patient_i + '/' + image_i)

print(f'Load X data: {len(X)}')
print(f'Load Y data: {len(Y)}')

# labels
map_file = '../Data/scATOMIC_Macro/bayes8_fraction_c2.csv'
y_label = y_map_list(Y, map_file, 'patient_id', 'tnbc')
print(f'Load Y label: {len(y_label)}')

# gene data process
gene_data = gene_process(data_gene, Y)
#gene_data['label'] = y_label

# images matrix
X_matrix = read_data(X, img_type='RGB',reshape_image=False)
print(f'Load X matrix: {X_matrix.shape}')

X_matrix = torch.tensor(np.array(X_matrix).transpose(0,3,1,2))
print(f'X matrix shape: {X_matrix.shape}')

gene_data = torch.tensor(gene_data.values)
print(f'gene data shape: {gene_data.shape}')

# Labels data reshape
y_label = torch.tensor(y_label)
print(f'y label shape: {y_label.shape}')

# load model

model = Combine_Model_SMLP(gene_data.shape[1],2)

model.load_state_dict(torch.load('../Results/net_save.pth'))

#  Results of Grad-CAM

for i in range(X_matrix.shape[0]):
    #print(int(y_label[i]))

    x_all = [X_matrix[i,:,:,:].unsqueeze(dim=0).float(), gene_data[i].unsqueeze(dim=0).float()]
    main(model, turn_images(X_matrix[i,:,:,:].unsqueeze(dim=0)), 
         x_all, int(y_label[i]),i)

