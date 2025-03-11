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

from utils import *
from model import *

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
map_file = '../Data/scATOMIC_Macro/CAF_labels.csv'
y_label = y_map_list(Y, map_file, 'patient_id', 'labels')
print(f'Load Y label: {len(y_label)}')

# gene data process
gene_data = gene_process(data_gene, Y)
#gene_data['label'] = y_label

# images matrix
X_matrix = read_data(X, img_type='RGB',reshape_image=False)
print(f'Load X matrix: {X_matrix.shape}')

# Images data reshape
X_matrix = torch.tensor(np.array(X_matrix).transpose(0,3,1,2))
print(f'X matrix shape: {X_matrix.shape}')

gene_data = torch.tensor(gene_data.values)
print(f'gene data shape: {gene_data.shape}')
# Labels data reshape
y_label = torch.tensor(y_label)
print(f'y label shape: {y_label.shape}')

# Dataloader
num_epoch = 50
batch_size = 16

# model
model = Combine_Model_SMLP(gene_data.shape[1],2)

learning_rate = 0.01

criterion_s = nn.CrossEntropyLoss()
opt_model = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

# five fold cross validation
lenth = len(X_matrix)
pot = int(lenth / 5)
print('lenth', lenth)
print('pot', pot)

random_num = random.sample(range(0, lenth), lenth)
#print(random_num)
for i_time in range(5):

    test_num = random_num[pot * i_time:pot * (i_time + 1)]
    train_num = random_num[:pot * i_time] + random_num[pot * (i_time + 1):]

    X1_train = X_matrix[train_num]
    X1_test = X_matrix[test_num]

    X2_train = gene_data[train_num]
    X2_test = gene_data[test_num]

    y_train = y_label[train_num]
    y_test = y_label[test_num]

    torch_dataset = Data.TensorDataset(X1_train.float(), X2_train.float(), y_train)
    train_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    test_dataset = Data.TensorDataset(X1_test.float(), X2_test.float(), y_test)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

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
            print(y_test_pred)
            
            y_test = test_y.to(torch.int64)

            loss_test = criterion_s(input=y_test_pred, target=y_test)
            acc_test = acc_multi(y_test_pred, y_test,classes = 2)
            acc_test_all.append(acc_test)
              
                     
        print('All | Epoch: ',epoch,'|','loss train: ',loss_train.item(),'acc train: ',mean_score(acc_train_all),'|',
            'loss test: ',loss_test.item(),'acc test: ',mean_score(acc_test_all))
    
    print('========================')
    
        

# Save model
path_net = '../Results/net_save.pth'
torch.save(model.state_dict(), path_net)
