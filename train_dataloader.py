'''
Team 17 - CS 7641 ML Project Fall 2021
"Visual Reasoning for the Visually Impaired"
Authors: Angana Borah, Devshree Bharatia, Dimitri Adhikary, Megha Thukral, Yusuf Ali

Script Summary: Perform training and evaluation on the filtered VizWiz dataset
'''

import copy
import glob
import os
import argparse

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
    '''
    Function which returns the average loss value for all the samples in a training/validation batch based on the paper: https://arxiv.org/pdf/1708.00584.pdf
    Implementation taken from the PyTorch VizWiz GitHub website: https://github.com/DenisDsh/VizWiz-VQA-PyTorch/

    Args:
        MLP_output: Tensor of size (batch_size, 3000) - batch output of final layer of the MLP netowrk
        local_labels: Tensor of size (batch_size,3000) - batch of label vectors for corresponding MLP output 
        LEN_TRAIN_SET: int - size of training set (default value = 0 ---> so if calculate_loss() is called from evaluate(), the value of LEN_TRAIN_SET = 0 by default)
        LEN_VAL_SET: int - size of validation set (default value = 0 ---> so if calculate_loss() is called from train(), the value of LEN_VAL_SET = 0 by default)

    Returns:
        loss: float - average loss value for the batch
    '''

    log_softmax = nn.LogSoftmax(dim = 1).cuda()

    nll = -1 * log_softmax(MLP_output)                              # calculate log softmax of the MLP_output
    loss = (nll * local_labels / 10).sum(dim = 1).mean()            # calculate soft cross-entropy loss as defined in https://arxiv.org/pdf/1708.00584.pdf

    if LEN_TRAIN_SET != 0:                                          # if condition to check whether the loss function was called from train() or evaluate() and apply normalistion based on that
        loss = loss / LEN_TRAIN_SET                                 # Normalise the training loss

    elif LEN_VAL_SET != 0:
        loss = loss / LEN_VAL_SET                                   # Normalise the validation loss
    
    else:
        print("\n Cannot calculate loss !")

    return loss



def train(epoch, train_loader, model, optimizer, tb, GPU_DEVICE, LEN_TRAIN_SET):
    '''
    Perform training on the training dataset and save loss values for tensorboard visualisation if required

    Args:
        epoch: index of the current epoch (int)
        train_loader: training set dataloader (PyTorch dataloader)
        model: MLP network (derived class torch.nn.Module)
        optimizer: Adam optimiser (torch.optim.Adam instance)
        tb: TensorBoard summary writer object
        GPU_DEVICE: name of GPU device
        LEN_TRAIN_SET: size of training set
    Return:
        None
    '''
    
    with tqdm(train_loader, unit="batch") as tepoch_train:          # tqdm wrapper for visualisation in command terminal 

        model.train()                                               # set the model into "training" mode

        train_loss = 0   
        
        for local_batch, local_labels in tepoch_train:
            
            if torch.cuda.is_available():                           # Shift the batched data and labels onto GPU for faster computation, if GPU is available 
                local_batch, local_labels = local_batch.to(GPU_DEVICE), local_labels.to(GPU_DEVICE)
            
            local_batch = local_batch.float()                       # Convert the batched data and labels to float type as it is required for gradient computation
            local_labels = local_labels.float()
            
            local_batch = local_batch.requires_grad_(True)          # Ensure batched data and labels have associated gradient tensors which will be updated in the backward propation step
            local_labels = local_labels.requires_grad_(True)

            optimizer.zero_grad()                                   # zero out the gradients (for more details refer to : https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)  
            
            MLP_output = model(local_batch)                         # forward pass of the SimpleNet() MLP network

            loss = calculate_loss(MLP_output, local_labels, LEN_TRAIN_SET)  # calculate train loss 

            loss.requires_grad_(True)                               # Ensure loss variable has an associated gradient tensor so that it can be used in the backprop step and is appended to the dynamic computation graph used by PyTorch for auto-differentiation
            
            loss.backward()                                         # Perform backprop 

            optimizer.step()                                        # Update the parameters based on Adam optimiser update rule

            # acc = vqa_accuracy(out.data, a.data).cpu()

            train_loss += loss.item()
        
        print("\n Epoch ", epoch + 1, " Train Loss : ", train_loss)
    
    if tb is not None:    
        tb.add_scalar('Train_Loss', train_loss, epoch)              # Save the train loss value for current epoch for TensorBoard visualisation



def evaluate(epoch, val_loader, model, tb, GPU_DEVICE, LEN_VAL_SET):
    '''
    Perform evaluation of teh trained model on the validation dataset and save loss values for tensorboard visualisation if required

    Args:
        epoch: index of the current epoch (int)
        val_loader: validation set dataloader (PyTorch dataloader)
        model: MLP network (derived class torch.nn.Module)
        tb: TensorBoard summary writer object
        GPU_DEVICE: name of GPU device
        LEN_VAL_SET: size of validation set
    Return:
        None
    '''
    
    with tqdm(val_loader, unit="batch") as tepoch_val:              # tqdm wrapper for visualisation in command terminal 

        model.eval()                                                # set the model into "evaluation" mode

        val_loss = 0   
        
        with torch.no_grad():

            for local_batch, local_labels in tepoch_val:
            
                if torch.cuda.is_available():                       # Shift the batched data and labels onto GPU for faster computation, if GPU is available 
                    local_batch, local_labels = local_batch.to(GPU_DEVICE), local_labels.to(GPU_DEVICE)
            
                local_batch = local_batch.float()                   # Convert the batched data and labels to float type as it is required for gradient computation
                local_labels = local_labels.float()
                
                MLP_output = model(local_batch)                     # forward pass of the SimpleNet() MLP network

                loss = calculate_loss(MLP_output, local_labels, LEN_VAL_SET)    # calculate validation loss 

                val_loss += loss.item()
            
            print("\n Epoch ", epoch + 1, " Val Loss : ", val_loss)
        
        if tb is not None:    
            tb.add_scalar('Val_Loss', val_loss, epoch)               # Save the validation loss value for current epoch for TensorBoard visualisation



def vqa_accuracy(predicted, true):
    '''
    Copied the function from the VizWiz PyTorch GitHub website: https://github.com/DenisDsh/VizWiz-VQA-PyTorch/
    Not sure if this is correct - so not using this function right now
    '''

    print("\n Preicted Shape: ", predicted.shape)
    print("\n True Shape: ", true.shape)

    # _, predicted_index = predicted.max(dim=1, keepdim=True)
    
    # agreeing = true.gather(dim=1, index=predicted_index)

    _, predicted_index = torch.max(predicted)
    
    agreeing = true.gather(dim=1, index = predicted_index)
    
    return (agreeing * 0.33333).clamp(max=1)



def main(NUM_EPOCHS, BATCH_SIZE, LEARNING_RATE, TRAIN_SAMPLES, VAL_SAMPLES, DATA_PATH, SAVE_TENSORBOARD_VAL):
    '''
    Main function which detects GPU, loads VizWiz PyTorch dataset and dataloaders, performs training and evaluation and saves loss values for tensorboard visuaisation (if required)

    Args:
        NUM_EPOCHS: number of epochs to train for (int)
        BATCH_SIZE: number of data samples in one batch of training (int)
        LEARNING_RATE: learnign rate for Adam optimsier (float)
        TRAIN_SAMPLES: size of training set (int)
        VAL_SAMPLES: size of validation set (int)
        DATA_PATH: absolute path of pkl_files directory (str) 
        SAVE_TENSORBOARD_VAL: flag variable which decides if tensorbaord visualisation is required or not (str: 'yes'/ 'no')

    Returns:
        None
    '''

    print("\n Is GPU Available: ", torch.cuda.is_available())                           # Check if GPU is available
    
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        print("\n Device Name: ", device)
        GPU_DEVICE = device                                                             # Assign identified GPU device to "GPU_DEVICE" variable which will be used to transfer computation to GPU
    
    if SAVE_TENSORBOARD_VAL == 'yes':
        SAVE_TENSORBOARD_VAL = True                                                     # "SAVE_TENSORBOARD_VAL" is flag variable which determiens if train/val losses will be saved for visualisation or not
    else:
        SAVE_TENSORBOARD_VAL = False

    # -------------------------------------------------------------------------------------------------------------------
    # Setup PyTorch Dataset and Dataloader from the filtered VizWiz Dataset

    print("\n Loading the Dataset ...")

    VizWiz_dataset = dataset.VizWiz_VQA_Dataset(DATA_PATH)                              # Load the PyTorch dataset (refer to dataset.py for more details) by loading the pkl files from the directory entered in the terminal

    print("\n Dataset Loaded !")

    print("\n Creating Training and Validation splits ...")
    
    train_set, val_set = random_split(VizWiz_dataset, [TRAIN_SAMPLES, VAL_SAMPLES])     # Randomly split the loaded dataset into training set and validation | No.of samples in each set is specified in command terminal while running the script 

    print("\n Training and Validation splits created !")

    print("\n Creating PyTorch dataloaders for training and validation sets ...")

    train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)          # Create training PyTorch dataloader (for setting up batched_data) using the training set loaded earlier 

    val_loader = DataLoader(val_set, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)              # Create validation PyTorch dataloader (for setting up batched_data) using the validation set loaded earlier 

    print("\n PyTorch dataloaders for training and validation sets created !")
    
    LEN_TRAIN_SET = len(train_set)                                                                          # Number of samples in training set
    LEN_VAL_SET = len(val_set)                                                                              # Number of samples in validation set

    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
    # Instantiate the SimpleNet() MLP fully connected network, Adam optimiser 

    print("\n Setting up Deep Network...")

    model = models.SimpleNet()

    print("\n Deep Network instantiated !")
    
    if torch.cuda.is_available():
        model.to(GPU_DEVICE)                                                                                # Transfer model to gpu for fast computation, if GPU is available 
        print("\n Model transferred to GPU !")
    
    else: 
        print("\n Model attached to CPU as GPU not found !")

    print("\n Setting up Adam optimiser...")

    optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

    print("\n Optimiser instantiated !")
    
    # ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    # Perform training and save epoch loss values for tensorboard visualisation (if required)
    
    if SAVE_TENSORBOARD_VAL == True:
        tb = SummaryWriter()                                                                                # Create the tensorbaord "Summary Writer" object which stores train and val losses for visualisation 
    else:
        tb = None

    print("\n Starting traing and evaluation...")
    
    for epoch in (range(NUM_EPOCHS)):     
        
        train(epoch, train_loader, model, optimizer, tb, GPU_DEVICE, LEN_TRAIN_SET)                         # Perform training

        evaluate(epoch, val_loader, model, tb, GPU_DEVICE, LEN_VAL_SET)                                     # Perform validation

    if SAVE_TENSORBOARD_VAL == True:
        tb.close()
    
    torch.cuda.empty_cache()                                                                                # Clear the GPU cache

    print("\n Training and Validation done !")

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--num_epochs', required = True, type = int, help = "Enter the number of epochs to train for")
    parser.add_argument('--batch_size', required = True, type = int, help = "Enter the batch_size to be used for training")
    parser.add_argument('--learning_rate', required = True, type = float, help = "Enter the learning_rate to be used for training")
    parser.add_argument('--train_samples', required = True, type = int, help = "Enter number of samples used in train set out of 13108 (train_samples + val_samples = 13108)")
    parser.add_argument('--val_samples', required = True, type = int, help = "Enter number of samples used in val set out of 13108 (train_samples + val_samples = 13108)")
    parser.add_argument('--path_pkl_files', type = str, help = "Enter the absolute directory location of the pkl_files folder")
    parser.add_argument('--tensor_board_viz', type = str, help = "Is tensorboard visualisation required for train and val losses: Yes/No ")

    args = parser.parse_args()
    
    main(args.num_epochs, args.batch_size, args.learning_rate, args.train_samples, args.val_samples, args.path_pkl_files, args.tensor_board_viz)