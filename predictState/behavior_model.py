#!/usr/bin/env python
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from io import open
import os.path
from os import path
import random
import numpy as np

import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
from torch import nn

class LSTM(nn.Module):
    def __init__(self,input_size,output_size,hidden_dim,n_layers,x_shape):
        super(LSTM, self).__init__()
        self.hidden_dim=hidden_dim
        self.n_layers = n_layers

        # define an RNN with specified parameters
        self.lstm = nn.LSTM(input_size, hidden_dim,n_layers,dropout=0., batch_first=True,bidirectional=False)
        # # last, fully-connected layer
        self.fc1 = nn.Linear(hidden_dim,hidden_dim*2)
        self.fc = nn.Linear(hidden_dim*2, output_size)
        self.fc2 = nn.Linear(x_shape,output_size)
        
    def forward(self, x, hidden):
        batch_size = x.size(0)
        r_out, hidden = self.lstm(x, hidden)
        # get RNN outputs
        r_out = r_out.contiguous().view(-1, self.hidden_dim)  

        # get final output
        output = self.fc1(r_out)
        output = self.fc(output)
        output = output.view(batch_size,-1,5)
        output = output[:,-1]
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        weight = next(self.parameters()).data
        
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        if train_on_gpu:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device))    
        return hidden
    
if torch.cuda.is_available():
    train_on_gpu = True
else:
    train_on_gpu = False