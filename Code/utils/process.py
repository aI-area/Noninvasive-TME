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
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,auc,accuracy_score,r2_score
import cv2
import torch.utils.data as Data
from tqdm import tqdm

from PIL import Image
import matplotlib.pyplot as plt
import gc
import random

def load_data(path_all):
    data_all = pd.DataFrame()
    X = []
    Y = []

    for patient_i in os.listdir(path_all):
            
        for image_i in os.listdir(path_all+patient_i):
            Y.append(patient_i)
            X.append(path_all+ patient_i + '/' + image_i)

    print(f'Load X data: {len(X)}')
    print(f'Load Y data: {len(Y)}')

    data_all['patient_id'] = Y
    data_all['image'] = X

    return data_all


def data_split_file(path_all, data_gene, train_size):

    data_all = load_data(path_all)
    grouped = data_all.groupby('patient_id')  
      
    num_groups = len(grouped)
    random_idx = random.sample(range(0, num_groups), num_groups)
    #print(random_idx)
    pot = int(train_size * num_groups)

    train_num = random_idx[pot:]
    test_num = random_idx[:pot]
  
    train_patient_ids = pd.DataFrame(list(grouped.groups.keys())).iloc[train_num] 
    test_patient_ids = pd.DataFrame(list(grouped.groups.keys())).iloc[test_num]
    #print(train_patient_ids)

    # select data
    train_data = data_all[data_all['patient_id'].isin(train_patient_ids[0])]  
    test_data = data_all[data_all['patient_id'].isin(test_patient_ids[0])]

    print(f'Train data: {train_data.shape}')
    print(f'Test data: {test_data.shape}')

    #print(train_data)
    train_x = train_data['image']
    train_y = train_data['patient_id']
    test_x = test_data['image']
    test_y = test_data['patient_id']

    #train_gene = pd.DataFrame()
    train_gene = data_gene[train_y]
    train_gene.index = data_gene.index
    train_gene = train_gene.T

    #test_gene = pd.DataFrame()
    test_gene = data_gene[test_y]
    test_gene.index = data_gene.index
    test_gene = test_gene.T

    return train_x, train_gene, list(train_y), test_x, test_gene, list(test_y)



def y_map_list(y_list, map_file, map_name, map_label):
    data_map = pd.read_csv(map_file)
    map_list_name = np.array(data_map[map_name])
    map_list_label = np.array(data_map[map_label])
    
    y_map = []
    for i in range(len(y_list)):
        for j in range(len(map_list_name)):
            if y_list[i] == map_list_name[j]:
                y_map.append(map_list_label[j])
    
    return y_map


def read_data(filename, img_type, reshape_image):
    XX_all = []
    for i in filename:
    # 读取图像
    # print i
        
        # 将一个4通道（包含透明度参数alpha）转化为rgb三通道
        if img_type == "RGB":
            image = Image.open(i)
            image = image.convert("RGB")
        elif img_type == "GRAY":
            image = cv2.imread(i,0)
            
            
        img_array = np.array(image)
        
        # 图像像素大小一致
        dst_size = (224, 224)
        # 224*224*3
        img_resize = cv2.resize(img_array, dst_size, interpolation = cv2.INTER_AREA)
        if reshape_image:
            XX_all.append(((img_resize / 255).flatten()))
        else:
            XX_all.append(img_resize/255)

    return np.array(XX_all)

def acc(pred, labels):
    count = 0
    a = torch.where(pred >0.6, 1,0).type(torch.int32)
    b = labels.type(torch.int32)
    for i in torch.eq(a,b):
        if i:
            count += 1

    acc_all = count / len(pred)
    return acc_all

def acc_multi(pred, labels, classes):
    count = 0
    a = pred.view(-1,classes)
    #print(a)
    a = a.detach().numpy()
    
    
    a_all = torch.tensor(np.argmax(a, axis=1))
    #print(a_all)
    b = labels.type(torch.int32)
    #print(b)
    for i in torch.eq(a_all,b):
        if i:
            count += 1

    acc_all = count / len(a_all)
    return acc_all

def r2(y_true, y_pred):
    y_true = y_true.detach().numpy()
    y_pred = y_pred.detach().numpy()

    return r2_score(y_true, y_pred)


def mean_score(x):
    return sum(x) / len(x)

def max_score(x):
    return max(x)



def gene_process(data_gene, X_name):
    list_all = pd.DataFrame()
    list_all = data_gene[X_name]
    list_all.index = data_gene.index
    gene_list = list_all.T

    return gene_list

def load_feature_importance(model, mark):
    model.eval()
    lst = []
    df = pd.DataFrame()
    for name, param in model.named_parameters():
        if len(param.detach()) == 1 and mark in name:
            weight = float(param.detach().cpu())
            if weight < 0:
                lst.append(0)
            else:
                lst.append(weight)
    S = sum(lst)
    df["weight_all"] = lst
    for t in range(len(lst)):
        lst[t] = lst[t]/S
    #df["index"] = [1, 2, 3]
    df["weight"] = lst
    print(df)
    df.to_excel(r"%s.xlsx"%mark,index=True)


def onehot2label(hot_list, classes):
    
    hot_list = hot_list.view(-1, classes)
  
    hot_list = hot_list.detach().numpy()
    label_list = np.argmax(hot_list, axis=1)
    return label_list


def turn_images(input_images):

    input_images_np = input_images.cpu().data.numpy().copy()
    
    x_array = np.squeeze(input_images_np)
    
    x_array = x_array*255
    
    slice_x = x_array
    slice_x = np.transpose(slice_x, (1,2,0))

    return slice_x


def save_image(save_dir, input_images, output_images, idx):


    input_images_np = input_images.cpu().data.numpy().copy()
    output_images_np = output_images.cpu().data.numpy().copy()
    #real_images_np = real_images.cpu().data.numpy().copy()

    #y_array = np.squeeze(real_images_np)
    img_arr = np.squeeze(output_images_np)
    x_array = np.squeeze(input_images_np)
    #print(x_array.shape)

    #y_array = y_array*255
    img_arr = img_arr*255
    x_array = x_array*255
    #print('img_arr.shape',img_arr.shape)

    for i in range(img_arr.shape[0]):
    
        slice_i = img_arr[i,:,:]
        #slice_y = y_array[i,:,:]
        slice_x = x_array[i,:,:,:]
        #print('slice_s.shape',slice_x.shape)

        slice_x = np.transpose(slice_x, (1,2,0))

        slice_x = slice_x.astype(np.uint8)

        img_x = Image.fromarray(slice_x)


        slice_i = slice_i.astype(np.uint8)
        img_pic = Image.fromarray(slice_i)

    
        plt.subplot(1,2,1)
        plt.imshow(img_x)
        plt.axis("off")
        plt.subplot(1,2,2)
        plt.imshow(img_pic, cmap='gray')
        plt.axis("off")

        plt.savefig('%s/%s_%s_val.png'%(save_dir,idx,i), bbox_inches="tight", pad_inches=0)

        plt.clf()
        plt.close()

def save_image_4(save_dir, input_images, output_images, i):


    input_images_np = input_images.cpu().data.numpy().copy()
    output_images_np = output_images.cpu().data.numpy().copy()
    #real_images_np = real_images.cpu().data.numpy().copy()

    #y_array = np.squeeze(real_images_np)
    img_arr = np.squeeze(output_images_np)
    x_array = np.squeeze(input_images_np)
    #print(x_array.shape)

    

    #y_array = y_array*255
    img_arr = img_arr*255
    x_array = x_array*255
    #print('img_arr.shape',img_arr.shape)

    
    
    slice_i = img_arr
    #slice_y = y_array[i,:,:]
    slice_x = x_array
    #print('slice_s.shape',slice_x.shape)

    slice_x = np.transpose(slice_x, (1,2,0))

    slice_x = slice_x.astype(np.uint8)

    img_x = Image.fromarray(slice_x)

    #slice_y = slice_y.astype(np.uint8)
    #img_y = Image.fromarray(slice_y)
    #print(slice_i.shape)
    slice_i = slice_i.astype(np.uint8)
    img_pic = Image.fromarray(slice_i)


    plt.subplot(1,2,1)
    plt.imshow(img_x)
    plt.axis("off")
    plt.subplot(1,2,2)
    plt.imshow(img_pic, cmap='gray')
    plt.axis("off")
    #plt.subplot(1,3,3)
    #plt.imshow(img_y, cmap='gray')
    #plt.axis("off")

    plt.savefig('%s/%s_save.png'%(save_dir, i), bbox_inches="tight", pad_inches=0)

    plt.clf()
    plt.close()

def save_image_1(output_images, save_dir):

    output_images_np = output_images.cpu().data.numpy().copy()

    img_arr = np.squeeze(output_images_np)

    img_arr = img_arr*255
    #print(img_arr.shape)
    
    slice_i = img_arr

    slice_i = slice_i.astype(np.uint8)
    img_pic = Image.fromarray(slice_i)

    plt.imshow(img_pic, cmap='gray')
    plt.axis("off")
    plt.savefig(save_dir, bbox_inches="tight", pad_inches=0)
    
    plt.clf()
    plt.close()

def grad2image(image_path, save_path, name):
    image = cv2.imread(image_path)  
    
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)  
    
    kernel = np.ones((3,3), np.uint8)  

    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  
    

    img_gray = cv2.cvtColor(opening, cv2.COLOR_BGR2GRAY)
    threshold = 200
    threshold = np.mean(img_gray)
    img_gray[img_gray>threshold] = 255
    img_gray[img_gray<=threshold] = 0

    plt.imshow(img_gray, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path+'%s.jpg'%name, bbox_inches="tight", pad_inches=0)


def plot_roc_curve(y, scores, save_path):
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=2)

    auc = metrics.auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.savefig(save_path+'plot_AUC.pdf')
    #plt.show()