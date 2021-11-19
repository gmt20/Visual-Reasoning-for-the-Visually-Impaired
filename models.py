import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence


class SimpleNet(nn.Module):
    '''Simple Network with atleast 2 conv2d layers and two linear layers.'''

    def __init__(self):
        '''
        Init function to define the layers and loss function

        Note: Use 'sum' reduction in the loss_criterion. Read Pytorch documention to understand what it means

        Hints:
        1. Refer to https://pytorch.org/docs/stable/nn.html for layers
        2. Remember to use non-linearities in your network. Network without
        non-linearities is not deep.
        '''
        super().__init__()

        self.linear_1 = torch.nn.Linear(5090, 5000)
        self.linear_2 = torch.nn.Linear(5000, 4500)
        self.linear_3 = torch.nn.Linear(4500, 4000)
        self.linear_4 = torch.nn.Linear(4000, 3500)
        self.linear_5 = torch.nn.Linear(3500, 3000)


    def forward(self, combined):

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