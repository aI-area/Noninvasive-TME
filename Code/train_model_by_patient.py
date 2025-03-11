import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
import os

from torch.autograd import Variable
from torch.autograd import Function
from torchvision import models, transforms
from torchvision import utils

import sys
import argparse

import torch.nn.init as init
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,accuracy_score
import cv2
import torch.utils.data as Data
from tqdm import tqdm

#from PIL import Image
#import matplotlib.pyplot as plt
import gc

from process import *
from model import *
from Grad_CAM_module import *
# Load data
# gene data
data_gene = pd.read_csv("../Data/breast_expr_c.csv",index_col=0)
data_gene = data_gene.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))

path_all = 'G:/BreastImages/TCIA/BRCA_select/'

train_x, train_gene, train_y, test_x, test_gene, test_y = data_split_file(path_all, data_gene, 0.3)


X_matrix_train = read_data(train_x, img_type='RGB',reshape_image=False)
X_matrix_test = read_data(test_x, img_type='RGB',reshape_image=False)
print('Train X matrix shape: ', X_matrix_train.shape)
print('Test X matrix shape: ', X_matrix_test.shape)

# labels
map_file = '../Data/CAF_labels.csv'

y_train = y_map_list(train_y, map_file, 'patient_id', 'labels')
y_test = y_map_list(test_y, map_file, 'patient_id', 'labels')

print(f'Load Y label: {len(y_train)}')
print(f'Load Y label: {len(y_test)}')

# Images data reshape
X1_train = torch.tensor(np.array(X_matrix_train).transpose(0,3,1,2))
X1_test = torch.tensor(np.array(X_matrix_test).transpose(0,3,1,2))

print(f'Train X1: {X1_train.shape}')
print(f'Test X1: {X1_test.shape}')

X2_train = torch.tensor(train_gene.values)
X2_test = torch.tensor(test_gene.values)

print(f'Train X2: {X2_train.shape}')
print(f'Test X2: {X2_test.shape}')

# Labels data reshape
y2_train = torch.tensor(y_train)
y2_test = torch.tensor(y_test)

print(f'Train Y2: {y2_train.shape}')
print(f'Test Y2: {y2_test.shape}')

# Dataloader
num_epoch = 50

batch_size = 16

torch_dataset = Data.TensorDataset(X1_train.float(), X2_train.float(), y2_train)
train_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)

test_dataset = Data.TensorDataset(X1_test.float(), X2_test.float(), y2_test)
test_loader = Data.DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
)
# model
model = Combine_Model(X2_train.shape[1],2)
#model = Combine_Model_SMLP(X2_train.shape[1],2)

'''learning_rate = 0.01

criterion_s = nn.CrossEntropyLoss()
opt_model = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

for epoch in range(num_epoch):
    acc_train_all = []
    

    # train    
    for batch_x1, batch_x2, batch_y in train_loader:
        model.train()
        
        batch_x = [batch_x1.float(), batch_x2.float()]
        y_train_pred = model(batch_x, gene = True)        
   
        batch_y = batch_y.to(torch.int64)
   
        loss_train = criterion_s(input=y_train_pred, target=batch_y)
        acc_train = acc_multi(y_train_pred, batch_y.squeeze(), classes = 2)
        acc_train_all.append(acc_train)
      

        opt_model.zero_grad()
        loss_train.requires_grad_(True)
        loss_train.backward()
        opt_model.step()

        # test
        acc_test_all = []
        for test_x1, test_x2, test_y in test_loader:
            model.eval()

            test_x = [test_x1.float(), test_x2.float()]
            y_test_pred = model(test_x, gene=True)
            
            y_test = test_y.to(torch.int64)

            loss_test = criterion_s(input=y_test_pred, target=y_test)
            acc_test = acc_multi(y_test_pred, y_test,classes = 2)
            acc_test_all.append(acc_test)
        print('| Epoch: ',epoch,'|','loss train: ',loss_train.item(),'acc train: ',mean_score(acc_train_all),'|',
          'loss test: ',loss_test.item(),'acc test: ',mean_score(acc_test_all))
            
            
            
    print('All | Epoch: ',epoch,'|','loss train: ',loss_train.item(),'acc train: ',mean_score(acc_train_all),'|',
          'loss test: ',loss_test.item(),'acc test: ',mean_score(acc_test_all))
       

# 保存模型
path_net = 'D:/1.June/singlecell/Bayes_CNN/Data/net_bcell.pth'
torch.save(model.state_dict(), path_net)
'''

model.load_state_dict(torch.load('D:/1.June/singlecell/Bayes_CNN/Results/model/net_CAF_bayes_n_p.pth'))

def main(model, img, X_transform, target_category, i):
    #model = models.mobilenet_v3_large(pretrained=True)
    #target_layers = [model.features[-1]]
    #print(target_layers)
    
    model = model
    target_layers = [model.conv1.conv.conv]
    #target_layers = [model.conv1.conv.new_conv2]
  
    
    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=False)
    #target_category = 1
    # target_category = 254  # pug, pug-dog
    
    
    #grayscale_cam = cam(X_transform, X1_gene.float(), target_category)
    

    grayscale_cam = cam(X_transform, target_category)

    grayscale_cam = grayscale_cam[0, :]
    visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                      grayscale_cam,
                                      use_rgb=True)
    plt.imshow(visualization)
    plt.axis('off')
    plt.savefig('H:/segment_breast/MicroE/CAF_c_n_p/{}/segment_test_{}.png'.format(int(target_category),i))

    #plt.show()

#  Results of Grad-CAM

for i in range(X1_test.shape[0]):
    print(int(y2_test[i]))
    #print(float(y_label[i]))
 
    x_all = [X1_test[i,:,:,:].unsqueeze(dim=0).float(), X2_test[i].unsqueeze(dim=0).float()]
    #main(model, X_transform[i,:,:,:].unsqueeze(dim=1),int(target_category[i]))
    main(model, turn_images(X1_test[i,:,:,:].unsqueeze(dim=0)), 
         x_all, int(y2_test[i]),i)
    # 回归模型用 y_label[i].unsqueeze(dim=0)
    # 二分类模型用 int(y_label[i])
   
# Latent Embedding

output_test = torch.sigmoid(model.conv1(X1_test.float()))
#print(output_test.shape)
pd.DataFrame(output_test.detach().numpy()).to_csv('D:/1.June/singlecell/Bayes_CNN/Results/features/X_matrix_caf_c_n_p_test.csv', index=False)
pd.DataFrame(y2_test.detach().numpy()).to_csv('D:/1.June/singlecell/Bayes_CNN/Results/features/y_label_caf_c_n_p_test.csv', index=False)