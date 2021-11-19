'''
Team 17 - CS 7641 ML Project Fall 2021
"Visual Reasoning for the Visually Impaired"
Authors: Angana Borah, Devshree Bharatia, Dimitri Adhikary, Megha Thukral, Yusuf Ali

Script Summary: Define various deep network architectures which will be used for the VQA task
'''


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleNet(nn.Module):
    '''
    Simple Fully Connected Network (Multi Layer Perceptron) with 4 hidden fully connected linear layers
    '''

    def __init__(self):
        '''
        Init function to define the linear layers
        "torch.nn.Linear(A,B)" ---> defines a linear fully connected layer of input feature dimension "A" and output feature dimension "B"
        Number of hidden neurons in above linear layer = B (each neuron in hidden layer produces 1 output feature)

        Args: None
        Return: None
        '''

        super().__init__()                              # How to use super() ---> https://realpython.com/python-super/

        # Dimensions of output features (number of neurons) was set arbitrarily for each of the four hidden fully connected layers
        # Feel free to play around with the number of neurons in each layer and total number of layers 
        self.linear_1 = torch.nn.Linear(5090, 5000)     
        self.linear_2 = torch.nn.Linear(5000, 4500)
        self.linear_3 = torch.nn.Linear(4500, 4000)
        self.linear_4 = torch.nn.Linear(4000, 3500)
        self.linear_5 = torch.nn.Linear(3500, 3000)


    def forward(self, combined):
        '''
        Forward function which performs the forward pass on a single batch of input data
        
        Experimented with relu / tanh / sigmoid ---> relu helps to converge the loss function quickly within less number of epochs

        Observe that there is no non-linear activation after self.linear_5 ---> this is because we apply log_softmax on the output of the final layer (defined in "calculate_loss" function in train_dataloader.py)
        
        Args: combined ---> one batch of input data
        Return: output tensor of dimension (3000,) 
        '''

        out = self.linear_1(combined)
        out = F.relu(out)
        out = self.linear_2(out)
        out = F.relu(out)
        out = self.linear_3(out)
        out = F.relu(out)
        out = self.linear_4(out)
        out = F.relu(out)
        out = self.linear_5(out)

        return out