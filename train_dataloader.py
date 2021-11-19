import copy
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision
import torch.nn.functional as F

import matplotlib.pyplot as plt

import sys
import pickle as pk

import tensorflow as tf

import pickle, ast
import pandas as pd

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, random_split

from tqdm import tqdm

import models

import dataset


def calculate_loss(MLP_output, local_labels, LEN_TRAIN_SET = 0, LEN_VAL_SET = 0):
    
    log_softmax = nn.LogSoftmax(dim = 1).cuda()

    nll = -1 * log_softmax(MLP_output)
    loss = (nll * local_labels / 10).sum(dim = 1).mean()

    if LEN_TRAIN_SET != 0:
        loss = loss / LEN_TRAIN_SET

    elif LEN_VAL_SET != 0:
        loss = loss / LEN_VAL_SET
    
    else:
        print("\n Cannot calculate loss !")

    return loss



def train(epoch, train_loader, model, optimizer, tb, GPU_DEVICE, LEN_TRAIN_SET):

    log_softmax = nn.LogSoftmax(dim = 1).cuda()
    
    with tqdm(train_loader, unit="batch") as tepoch_train:

        model.train()

        train_loss = 0   
        
        for local_batch, local_labels in tepoch_train:
            
            local_batch, local_labels = local_batch.to(GPU_DEVICE), local_labels.to(GPU_DEVICE)
            
            local_batch = local_batch.float()
            local_labels = local_labels.float()
            
            local_batch = local_batch.requires_grad_(True)
            local_labels = local_labels.requires_grad_(True)

            optimizer.zero_grad()
            
            MLP_output = model(local_batch)

            loss = calculate_loss(MLP_output, local_labels, LEN_TRAIN_SET)

            loss.requires_grad_(True)
            
            loss.backward()

            optimizer.step()

            acc = vqa_accuracy(out.data, a.data).cpu()

            train_loss += loss.item()
        
        print("\n Epoch ", epoch + 1, " Train Loss : ", train_loss)
    
    if tb is not None:    
        tb.add_scalar('Train_Loss', train_loss, epoch)


def evaluate(epoch, val_loader, model, tb, GPU_DEVICE, LEN_VAL_SET):

    log_softmax = nn.LogSoftmax(dim = 1).cuda()
    
    with tqdm(val_loader, unit="batch") as tepoch_val:

        model.eval()

        val_loss = 0   
        
        with torch.no_grad():

            for local_batch, local_labels in tepoch_val:
            
                local_batch, local_labels = local_batch.to(GPU_DEVICE), local_labels.to(GPU_DEVICE)
            
                local_batch = local_batch.float()
                local_labels = local_labels.float()
                
                MLP_output = model(local_batch)

                loss = calculate_loss(MLP_output, local_labels, LEN_VAL_SET)

                val_loss += loss.item()
            
            print("\n Epoch ", epoch + 1, " Val Loss : ", val_loss)
        
        if tb is not None:    
            tb.add_scalar('Val_Loss', val_loss, epoch)



def vqa_accuracy(predicted, true):
    """ Approximation of VQA accuracy metric """

    print("\n Preicted Shape: ", predicted.shape)
    print("\n True Shape: ", true.shape)

    # _, predicted_index = predicted.max(dim=1, keepdim=True)
    
    # agreeing = true.gather(dim=1, index=predicted_index)

    _, predicted_index = torch.max(predicted)
    
    agreeing = true.gather(dim=1, index = predicted_index)
    
    return (agreeing * 0.33333).clamp(max=1)


def main():

    print("\n Is GPU Available: ", torch.cuda.is_available())
    
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print("\n Device Name: ", device)
        GPU_DEVICE = device

    # -------------------------------------------------------------------------------------------------------------------
    # Hyperparameter Definitions

    LEARNING_RATE = 1e-6
    NUM_EPOCHS = 3
    SAVE_TENSORBOARD_VAL = True
    BATCH_SIZE = 128
    TRAIN_SAMPLES = 10000
    VAL_SAMPLES = 3108

    # -------------------------------------------------------------------------------------------------------------------
    # Setup VizWiz Dataset

    data_path = '/home/yusuf/Desktop/Georgia Tech/Fall 2021/CS7641/project_midterm/'
    
    VizWiz_dataset = dataset.VizWiz_VQA_Dataset(data_path)

    train_set, val_set = random_split(VizWiz_dataset, [TRAIN_SAMPLES, VAL_SAMPLES])
    print("\n Size of train dataset: ", len(train_set))
    print("\n Size of val dataset: ", len(val_set))
    print(" \n Shape of train dataset: ", train_set[0][0].shape, train_set[0][1].shape)
    print(" \n Shape of val dataset: ", val_set[0][0].shape, val_set[0][1].shape)

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    print("\n Train Data Shapes: ", next(iter(train_loader))[0].shape, next(iter(train_loader))[1].shape)

    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)
    print("\n Val Data Shapes: ", next(iter(val_loader))[0].shape, next(iter(val_loader))[1].shape)

    LEN_TRAIN_SET = len(train_set)
    LEN_VAL_SET = len(val_set)

    # --------------------------------------------------------------------------------------------------------------------    
    # Define neural net, optimiser, and loss function 

    model = models.SimpleNet()
    model.to(GPU_DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)
    
    # --------------------------------------------------------------------------------------------------------------------
    # Perform training and save epoch loss values for tensorboard visualisation
    
    if SAVE_TENSORBOARD_VAL == True:
        tb = SummaryWriter()
    else:
        tb = None
    
    for epoch in (range(NUM_EPOCHS)):     
        
        train(epoch, train_loader, model, optimizer, tb, GPU_DEVICE, LEN_TRAIN_SET)

        evaluate(epoch, val_loader, model, tb, GPU_DEVICE, LEN_VAL_SET)

    if SAVE_TENSORBOARD_VAL == True:    
        tb.close()
    
    torch.cuda.empty_cache()

    
if __name__ == '__main__':
    main()