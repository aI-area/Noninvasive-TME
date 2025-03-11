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


class MLP(nn.Module):
    def __init__(self,in_feats,hid1_feats,hid2_feats,out_feats):
        super().__init__()

        self.mlp1 = nn.Linear(in_feats,hid1_feats)
        self.bn1 = nn.BatchNorm1d(hid1_feats)
        self.mlp2 = nn.Linear(hid1_feats,hid2_feats)
        self.bn2 = nn.BatchNorm1d(hid2_feats)
        self.mlp3 = nn.Linear(hid2_feats,out_feats)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.in_feats = in_feats
        self.softmax = nn.Softmax()
        

    def forward(self, x):

        x = x.view(-1,self.in_feats)

        #print(x.shape)
        out = self.mlp1(x)
        #out = self.MLP_attention(out)
        out = self.bn1(out)
        out = self.relu(out)
        #print(x)

        out = self.mlp2(out)
        #out = self.MLP_attention(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print(x.shape)

        out = self.mlp3(out)
        #out = self.MLP_attention(out)
        out = self.sigmoid(out)
        out = out.squeeze(-1)

        return out
    


class MLP_reg(nn.Module):
    def __init__(self,in_feats,hid1_feats,hid2_feats,out_feats):
        super().__init__()

        self.mlp1 = nn.Linear(in_feats,hid1_feats)
        self.bn1 = nn.BatchNorm1d(hid1_feats)
        self.mlp2 = nn.Linear(hid1_feats,hid2_feats)
        self.bn2 = nn.BatchNorm1d(hid2_feats)
        self.mlp3 = nn.Linear(hid2_feats,out_feats)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.in_feats = in_feats
        self.softmax = nn.Softmax()
        

    def forward(self, x):

        x = x.view(-1,self.in_feats)

        #print(x.shape)
        out = self.mlp1(x)
        #out = self.MLP_attention(out)
        out = self.bn1(out)
        out = self.relu(out)
        #print(x)

        out = self.mlp2(out)
        #out = self.MLP_attention(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print(x.shape)

        out = self.mlp3(out)
        #out = self.MLP_attention(out)
        #out = self.sigmoid(out)
        out = out.squeeze(-1)

        return out


# -*- coding: utf-8 -*-
# pytorch SMLP with only one hidden layer for multiclass classification

class presingle(torch.nn.Module):
    def __init__(self):
        super(presingle, self).__init__()
        self.presingle = torch.nn.Linear(1, 1, bias=False) # bias are set to False
        torch.nn.init.ones_(self.presingle.weight)
    def forward(self, x):
        x = self.presingle(x)
        return x

# model definition
class SMLP(nn.Module):
    # define model elements
    def __init__(self, n_inputs, hid1_feats, hid2_feats, out_feats, mark):
        super(SMLP, self).__init__()
        self.presingle = [presingle() for i in range(n_inputs)]
        self.input = n_inputs
        for i in range(n_inputs):
            setattr(self, mark + str(i + 1), self.presingle[i])


        self.mlp1 = nn.Linear(n_inputs, hid1_feats)
        self.bn1 = nn.BatchNorm1d(hid1_feats)
        self.mlp2 = nn.Linear(hid1_feats,hid2_feats)
        self.bn2 = nn.BatchNorm1d(hid2_feats)
        self.mlp3 = nn.Linear(hid2_feats,out_feats)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.in_feats = n_inputs
        self.softmax = nn.Softmax()


    def forward(self, x):
        lst = []
        for n in range(self.input):
            lst.append(torch.nn.functional.relu(self.presingle[n](x[:, n].unsqueeze(1))))
        input_cat = torch.cat((lst), 1)

        x = x.view(-1,self.in_feats)
        #print(x.shape)
        out = self.mlp1(input_cat)
        #out = self.MLP_attention(out)
        out = self.bn1(out)
        out = self.relu(out)
        #print(x)

        out = self.mlp2(out)
        #out = self.MLP_attention(out)
        out = self.bn2(out)
        out = self.relu(out)
        #print(x.shape)

        out = self.mlp3(out)
        #out = self.MLP_attention(out)
        out = self.sigmoid(out)
        out = out.squeeze(-1)
        
        return out





############################ UNet ####################################

class Block(nn.Module):

    def __init__(self, in_channels, features):
        super(Block, self).__init__()

        self.features = features
        self.conv1 = nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )
        self.conv2 = nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding='same',
                        )

    def forward(self, input):
        x = self.conv1(input)
        x = nn.BatchNorm2d(num_features=self.features)(x)
        x = nn.ReLU(inplace=True)(x)
        x = self.conv2(x)
        x = nn.BatchNorm2d(num_features=self.features)(x)
        x = nn.ReLU(inplace=True)(x)

        return x


class UNetC(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNetC, self).__init__()

        features = init_features
        self.conv_encoder_1 = Block(in_channels, features)
        self.conv_encoder_2 = Block(features, features * 2)
        self.conv_encoder_3 = Block(features * 2, features * 4)
        self.conv_encoder_4 = Block(features * 4, features * 8)

        self.bottleneck = Block(features * 8, features * 16)

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.conv_decoder_4 = Block((features * 8) * 2, features * 8)
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.conv_decoder_3 = Block((features * 4) * 2, features * 4)
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.conv_decoder_2 = Block((features * 2) * 2, features * 2)
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Block(features * 2, features)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        conv_encoder_1_1 = self.conv_encoder_1(x)
        conv_encoder_1_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_1_1)

        conv_encoder_2_1 = self.conv_encoder_2(conv_encoder_1_2)
        conv_encoder_2_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_2_1)

        conv_encoder_3_1 = self.conv_encoder_3(conv_encoder_2_2)
        conv_encoder_3_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_3_1)

        conv_encoder_4_1 = self.conv_encoder_4(conv_encoder_3_2)
        conv_encoder_4_2 = nn.MaxPool2d(kernel_size=2, stride=2)(conv_encoder_4_1)

        bottleneck = self.bottleneck(conv_encoder_4_2)

        conv_decoder_4_1 = self.upconv4(bottleneck)
        conv_decoder_4_2 = torch.cat((conv_decoder_4_1, conv_encoder_4_1), dim=1)
        conv_decoder_4_3 = self.conv_decoder_4(conv_decoder_4_2)

        conv_decoder_3_1 = self.upconv3(conv_decoder_4_3)
        conv_decoder_3_2 = torch.cat((conv_decoder_3_1, conv_encoder_3_1), dim=1)
        conv_decoder_3_3 = self.conv_decoder_3(conv_decoder_3_2)

        conv_decoder_2_1 = self.upconv2(conv_decoder_3_3)
        conv_decoder_2_2 = torch.cat((conv_decoder_2_1, conv_encoder_2_1), dim=1)
        conv_decoder_2_3 = self.conv_decoder_2(conv_decoder_2_2)

        conv_decoder_1_1 = self.upconv1(conv_decoder_2_3)
        conv_decoder_1_2 = torch.cat((conv_decoder_1_1, conv_encoder_1_1), dim=1)
        conv_decoder_1_3 = self.decoder1(conv_decoder_1_2)
        

        return torch.sigmoid(self.conv(conv_decoder_1_3))
    



class NewModel(nn.Module):  
    def __init__(self):  
        super(NewModel, self).__init__()  
        self.conv = UNetC() 
        #self.new_conv1 = nn.Conv2d(1, 3, kernel_size=3, padding='same')  
        #self.new_conv2 = nn.Conv2d(3, 1, kernel_size=3, padding='same')

        # 加两层MLP层

        self.fc1 = nn.Linear(224*224, 5120)  
        self.fc2 = nn.Linear(5120, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.dropout = nn.Dropout(0.5)
     
    def forward(self, x):  
        #x = self.original_model(x)
        #x = torch.relu(self.new_conv1(x))
        #print(x.shape)
        #x = torch.relu(self.new_conv2(x))
        #print(x.shape)
        x = self.conv(x)
        
        x = x.view(-1, 224*224)  
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))  
        
        #print(x.shape)
        return x


class Combine_Model(nn.Module):
    def __init__(self, in_feats, out_feats):   
        super(Combine_Model, self).__init__()  
        self.conv1 = NewModel()
        
        self.mlp_gene = MLP(in_feats, 10240, 1024, 512)

        self.mlp_cnn = MLP(512, 128, 32, out_feats)
        self.mlp_all = MLP(1024, 128, 32, out_feats)
  
    def forward(self, X_data, gene=False):
        x_image = X_data[0]

        x_image1 = self.conv1(x_image)
        if gene:
            x_gene = X_data[1]
            x_gene1 = self.mlp_gene(x_gene)
        
            x_combind = torch.cat((x_image1,x_gene1),dim=1)
            #print(x_combind.shape)
        
            x_fc = self.mlp_all(x_combind)
        else:
            x_fc = self.mlp_cnn(x_image1)
        
        return x_fc



class Combine_Model_reg(nn.Module):
    def __init__(self, in_feats):   
        super(Combine_Model_reg, self).__init__()  
        self.conv1 = NewModel()
        
        self.mlp_gene = MLP(in_feats, 10240, 1024, 512)

        self.mlp_cnn = MLP_reg(512, 128, 32, 1)
        self.mlp_all = MLP_reg(1024, 128, 32, 1)
  
    def forward(self, X_data, gene=False):
        x_image = X_data[0]

        x_image1 = self.conv1(x_image)
        if gene:
            x_gene = X_data[1]
            x_gene1 = self.mlp_gene(x_gene)
        
            x_combind = torch.cat((x_image1,x_gene1),dim=1)
            #print(x_combind.shape)
        
            x_fc = self.mlp_all(x_combind)
        else:
            x_fc = self.mlp_cnn(x_image1)
        
        return x_fc


class Combine_Model_SMLP(nn.Module):
    def __init__(self, in_feat, out_feat):   
        super(Combine_Model_SMLP, self).__init__()  
        self.conv1 = NewModel()
        

        self.mlp_gene = SMLP(in_feat, 10240, 1024, 512, mark = 'presingle_gene_')

        self.mlp_cnn = SMLP(512, 128, 32, out_feat, mark = 'presingle_cnn_')
        self.mlp_all = SMLP(1024, 128, 32, out_feat, mark = 'presingle_all_')
    
    def forward(self, data, gene=False):
        x_image = data[0]
        x_image1 = self.conv1(x_image)
        if gene:
            x_gene = data[1]
            x_gene1 = self.mlp_gene(x_gene)
        
            #x_combind = x_image1 + x_gene1
            x_combind = torch.cat((x_image1,x_gene1),dim=1)
            #print(x_combind.shape)
        
            x_fc = self.mlp_all(x_combind)
        else:
            x_fc = self.mlp_cnn(x_image1)
        
        return x_fc
            