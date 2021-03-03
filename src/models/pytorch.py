import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class PytorchMultiClass(nn.Module):

    '''
    num_features as input parameter
      attributes:
        layer_1: fully-connected layer with 32 neurons
        layer_out: fully-connected layer with 4 neurons
        softmax: softmax function

      methods:
        forward() with inputs as input parameter, perform ReLU and DropOut on the fully-connected layer followed by the output layer with softmax
    '''

    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)

def get_device():
    '''
      Set model to use the device available
    '''
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device


class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...

    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensor
    y_tensor : Pytorch tensor
        Target tensor

    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """
        
    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)
    
    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]
        
    def __len__ (self):
        return len(self.X_tensor)
    
    def to_tensor(self, data):
        return torch.Tensor(ndata)