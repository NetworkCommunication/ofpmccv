# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
from os import path

import numpy as np
import unicodedata
import string
import re
import time
import math
import random
import pandas as pd
import scipy.signal
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
import torch.nn.functional as F

# 设置随机种子来实现可重复性
from dataPrepare8 import get_dataloader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(2)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

MAX_LENGTH = 100
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')

# Set sequence length
length = 90
predict_length = 30
# Get training set, test set, validation set and complete data set
Training_generator, Test, Valid, WholeSet = get_dataloader(128,length,predict_length)
print(len(WholeSet))

class SelfAttention(nn.Module):
    def __init__(self, input_dim, num_heads):
        super(SelfAttention, self).__init__()

        self.input_dim = input_dim
        self.num_heads = num_heads

        assert self.input_dim % num_heads == 0

        self.head_dim = input_dim // num_heads

        self.qs_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.ks_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.vs_layer = nn.Linear(input_dim, input_dim, bias=False)

        self.fc_layer = nn.Linear(input_dim, input_dim)

    def forward(self, inputs):
        """
        Args:
            inputs: A tensor of ``(batch_size, seq_length, input_dim)``.
        Returns:
            A tensor of ``(batch_size, seq_length, input_dim)``.
        """

        batch_size, seq_length, input_dim = inputs.size()

        qs = self.qs_layer(inputs).view(batch_size, seq_length, self.num_heads, self.head_dim)
        ks = self.ks_layer(inputs).view(batch_size, seq_length, self.num_heads, self.head_dim)
        vs = self.vs_layer(inputs).view(batch_size, seq_length, self.num_heads, self.head_dim)

        # Compute QK^T
        weights = torch.matmul(qs, ks.permute(0, 1, 3, 2))
        weights = weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        weights = F.softmax(weights, dim=-1)

        # Compute weighted sum of Vs
        attention = torch.matmul(weights, vs)

        reshaped = attention.view(batch_size, seq_length, self.num_heads * self.head_dim)
        output = self.fc_layer(reshaped)

        return output


# Define neural network structure
class NNPred(nn.Module):
    def __init__(self, input_size, output_size,hidden_size,batch_size,num_heads=4, dropout=0.5):
        super(NNPred, self).__init__()

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.num_heads = num_heads

        self.in2lstm = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False,batch_first=True,dropout =dropout)
        self.in2bilstm = nn.Linear(input_size, hidden_size)
        self.bilstm = nn.LSTM(hidden_size, hidden_size,num_layers=self.num_layers,bidirectional=False,batch_first=True,dropout =dropout)

        self.decoder_lstm = nn.LSTM(int(hidden_size/2), int(hidden_size/2), num_layers=self.num_layers, bidirectional=False,
                                    batch_first=True, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)

        self.self_atten = SelfAttention(hidden_size*2, num_heads)

        self.fc0 = nn.Linear(hidden_size,hidden_size*2)
        self.fc1 = nn.Linear(hidden_size*2,int(hidden_size/2))
        self.in2out = nn.Linear(input_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2) ,output_size)
        self.tanh = nn.Tanh()

        l2_reg_weight = 0.001
        self.fc2.weight.data = self.fc2.weight.data - l2_reg_weight * self.fc2.weight.data.norm(2)

    def forward(self, input):
        bilstm_out,_= self.bilstm(self.in2bilstm(input))
        lstm_out,_= self.lstm(self.in2lstm(input))
        out = self.tanh(self.fc0(lstm_out+bilstm_out))

        out = self.dropout(out)

        self.attention_output = self.self_atten(out)

        mlp_out = self.tanh(self.fc1(self.dropout(self.attention_output)))

        # Concatenate attention output with original input
        out = mlp_out + self.in2out(input)

        out, _ = self.decoder_lstm(out)

        # Apply a final output layer to generate model predictions
        output = self.fc2(out)

        return output

# training function
def trainIters(encoder, epoches,learning_rate=0.0001,print_every=7):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer,mode='min',factor=0.1,
patience=120,verbose=True,threshold=0.0001,threshold_mode='rel',cooldown=0,min_lr=0,eps=1e-015)
    criterion = nn.MSELoss()
    loss_min = np.inf
    train_losses =[]
    valid_losses =[]
    encoder.train()
    for e in range(1, epoches + 1):
        train_loss = []
        for batch_i,(local_batch, local_labels) in enumerate(Training_generator):
            encoder.zero_grad()
            local_batch = local_batch.to(device)
            local_labels = local_labels.to(device)

            predY = encoder(local_batch)
            # Calculate the difference between the model predicted value and the true value loss
            loss = criterion(predY,local_labels).to(device)
            loss.backward()
            encoder_optimizer.step()
            train_loss.append(loss.item())
            # Calculate the loss on the validation set and output the log
            if batch_i % print_every == 0:
                valid_loss = []
                encoder.eval()
                for x,y in Valid:
                    x,y = x.to(device),y.to(device)
                    predict = encoder(x)
                    loss_valid = criterion(predict,y)
                    valid_loss.append(loss_valid.item())
                # After calculating the loss, convert to training mode
                encoder.train()
                train_loss_mean = np.mean(train_loss)
                valid_loss_mean = np.mean(valid_loss)
                train_losses.append(train_loss_mean)
                valid_losses.append(valid_loss_mean)
                scheduler.step(valid_loss_mean)
                print("Epoch:{}/{},Step:{}/{}".format(e,epoches,batch_i,len(Training_generator)),
                  "Train_Loss:{},Valid_Loss: {}".format(train_loss_mean,valid_loss_mean))
                if valid_loss_mean < loss_min :
                    print("valid_loss decrease!!!save the model.")
                    loss_min = valid_loss_mean
                    torch.save(encoder.state_dict(),'model/attentionStatePredict.pt')

# Calculate the X and Y coordinates of each predicted trajectory in the pY array
def calcu_XY(predY):
    vels = predY[:,:,0:2]
    rst_xy = np.zeros(predY[:,:,0:2].shape)
    rst_xy[:,:-predict_length,:] = predY[:,:-predict_length,2:4]
    delta_t = 0.1

    for i in range(predict_length):
        a = (vels[:,-(predict_length-i),:] - vels[:,-(predict_length+1-i),:])/delta_t
        delta_xy = vels[:,-(predict_length-i),:]*vels[:,-(predict_length-i),:]-vels[:,-(predict_length+1-i),:]*vels[:,-(predict_length+1-i),:]
        delta_xy = delta_xy/(2*a)
        rst_xy[:,-(predict_length-i),:] = rst_xy[:,-(predict_length+1-i),:] + delta_xy

    return rst_xy

# Evaluate neural networks
def Eval_net(encoder,optmizer=False):
    n_trajectory_batch = 0
    loss = []
    MSE_pres = []
    MSE_rsts = []
    for local_batch, local_labels in Test:
        n_trajectory_batch = n_trajectory_batch + 1
        criterion = nn.MSELoss()
        local_batch = local_batch.to(device)
        local_labels = local_labels.to(device)

        predY = encoder(local_batch)
        test_loss = criterion(predY,local_labels)
        loss.append(test_loss.item())
        std = WholeSet.std.repeat(local_batch.shape[0],x.shape[1],1)
        std = std[:,:,:4].to(device)
        mn = WholeSet.mn.repeat(local_batch.shape[0],x.shape[1],1)
        mn = mn[:,:,:4].to(device)
        rg = WholeSet.range.repeat(local_batch.shape[0],x.shape[1],1)
        rg = rg[:,:,:4].to(device)
        predY = (predY*(rg*std)+mn).detach().cpu()
        pY = np.array(predY)
        local_labels = (local_labels*(rg*std)+mn).detach().cpu()
        Y = np.array(local_labels)
        pY[:,:-predict_length,:] = Y[:,:-predict_length,:]
        rst_xy = calcu_XY(pY)

        if n_trajectory_batch > 20:
            break
        # Evaluate the prediction performance of the model
        for i in range(1):
            real_predict = torch.from_numpy(Y[:,-predict_length-1:,2:4])
            rst_predict = torch.from_numpy(rst_xy[:,-predict_length-1:,:2])
            pre_predict = torch.from_numpy(pY[:,-predict_length-1:,2:4])
            MSE_pre = criterion(real_predict,pre_predict)
            MSE_rst = criterion(real_predict,rst_predict)
            MSE_pres.append(MSE_pre.item())
            MSE_rsts.append(MSE_rst.item())
            print("MSE pre:",MSE_pre.item())
            print("MSE RST:",MSE_rst.item())
        print('Test loss:',test_loss.item())
    print("mean MSE  pre:",np.mean(MSE_pres))
    print("mean MSE rst:",np.mean(MSE_rsts))
    print('average loss:',np.mean(loss))

# Used to test model prediction capabilities
def predict(model, test_load, n, optimizer=False):
    test = iter(test_load)
    x, y = next(test)
    x, y = next(test)
    x, y = x.to(device), y.to(device)
    predY = model(x)
    criterion = nn.MSELoss()
    # Calculate the loss between model predictions and true labels using the MSE loss function
    test_loss = criterion(predY, y)
    std = WholeSet.std.repeat(x.shape[0], x.shape[1], 1)
    # Keep only the standard deviation of the first 4 features (i.e. velx, vely, x and y)
    std = std[:, :, :4].to(device)
    mn = WholeSet.mn.repeat(x.shape[0], x.shape[1], 1)
    mn = mn[:, :, :4].to(device)
    rg = WholeSet.range.repeat(x.shape[0], x.shape[1], 1)
    rg = rg[:, :, :4].to(device)
    predY = (predY * (rg * std) + mn).detach().cpu()
    pY = np.array(predY)
    local_labels = (y * (rg * std) + mn).detach().cpu()
    Y = np.array(local_labels)
    pY[:, :-predict_length, :] = Y[:, :-predict_length, :]
    rst_xy = calcu_XY(pY)
    real_predict = torch.from_numpy(Y[:, -predict_length - 1:, 2:4])
    rst_predict = torch.from_numpy(rst_xy[:, -predict_length - 1:, :2])
    pre_predict = torch.from_numpy(pY[:, -predict_length - 1:, 2:4])
    MSE_pre = criterion(real_predict, pre_predict)
    MSE_rst = criterion(real_predict, rst_predict)
    print("MSE pre:", MSE_pre.item())
    print("MSE RST:", MSE_rst.item())
    print("Test Loss:", test_loss.item())

def Eval_net_behavior(encoder,optimization=False):
    n_trajectory_batch = 0
    loss = []
    MSE_pres = []
    MSE_rsts = []
    accuracies = []
    for local_batch, local_labels in Test:
        n_trajectory_batch = n_trajectory_batch + 1
        criterion = nn.MSELoss()
        local_batch = local_batch.to(device)
        local_labels = local_labels.to(device)

        yl = local_batch[:,:,-1].view(-1,1)
        local_batch = compute_label(local_batch)
        pred = local_batch[:,:,-1].view(-1,1)
        equal = yl.squeeze()==pred.squeeze()
        accu = torch.mean(equal.type(torch.FloatTensor)).item()
        accuracies.append(accu)
        predY = encoder(local_batch)
        test_loss = criterion(predY,local_labels)
        loss.append(test_loss.item())
        std = WholeSet.std.repeat(local_batch.shape[0],x.shape[1],1)
        std = std[:,:,:4].to(device)
        mn = WholeSet.mn.repeat(local_batch.shape[0],x.shape[1],1)
        mn = mn[:,:,:4].to(device)
        rg = WholeSet.range.repeat(local_batch.shape[0],x.shape[1],1)
        rg = rg[:,:,:4].to(device)
        predY = (predY*(rg*std)+mn).detach().cpu()
        pY = np.array(predY )
        local_labels = (local_labels*(rg*std)+mn).detach().cpu()
        Y = np.array(local_labels)
        pY[:,:-predict_length,:] = Y[:,:-predict_length,:]
        rst_xy = calcu_XY(pY)
        if n_trajectory_batch > 20:
            break
        for i in range(1):
            real_predict = torch.from_numpy(Y[:,-predict_length-1:,2:4])
            rst_predict = torch.from_numpy(rst_xy[:,-predict_length-1:,:2])
            pre_predict = torch.from_numpy(pY[:,-predict_length-1:,2:4])
            MSE_pre = criterion(real_predict,pre_predict)
            MSE_rst = criterion(real_predict,rst_predict)
            print("MSE pre:",MSE_pre.item())
            print("MSE RST:",MSE_rst.item())
            MSE_pres.append(MSE_pre)
            MSE_rsts.append(MSE_rst)
        print('Test loss:',test_loss.item())
    print("mean MSE  pre:",np.mean(MSE_pres))
    print("mean MSE rst:",np.mean(MSE_rsts))
    print('average loss:',np.mean(loss))
    print("mean behavior accuracy:",np.mean(accuracies))

from behavior_model import LSTM

# Calculate the label corresponding to input_x
def compute_label(input_x):
    std = WholeSet.std.repeat(input_x.shape[0],x.shape[1],1)
    std = std.to(device)
    mn = WholeSet.mn.repeat(input_x.shape[0],x.shape[1],1)
    mn = mn.to(device)
    rg = WholeSet.range.repeat(input_x.shape[0],x.shape[1],1)
    rg = rg.to(device)
    print(input_x.shape)
    # Denormalize the input and transfer it to the CPU
    input_return = (input_x*(rg*std)+mn).detach().cpu()
    # Initialize an LSTM model
    net_behavior = LSTM(1,5,256,2,26)
    net_behavior.load_state_dict(torch.load('model/behavior_prediction.pth'))
    net_behavior.to(device)
    inputs = torch.from_numpy(input_return[:,:,:-1].detach().cpu().numpy().astype(np.float32)).to(device)
    inputs = inputs.view(-1,inputs.shape[2]).unsqueeze(2)
    h = net_behavior.init_hidden(inputs.shape[0])
    h = tuple([each.data for each in h])
    # Perform a softmax operation on the behavior to obtain the probability of each category.
    # Take the category with the highest probability as the output label
    behavior,h = net_behavior(inputs,h)
    _, class_ = torch.max(behavior, dim=1)
    class_ = torch.from_numpy(class_.detach().cpu().numpy().astype(np.double)).to(device)
    class_ = class_.view(input_x.shape[0],x.shape[1],-1)
    y_ = input_return[:,:,-1].view(-1,1).numpy().tolist()
    std = std[:,:,-1].unsqueeze(2)
    rg = rg[:,:,-1].unsqueeze(2)
    mn = mn[:,:,-1].unsqueeze(2)
    # Standardized processing
    class_ = (class_ -mn) / (std*rg)
    py = torch.cat((input_x[:,:,:-1],class_),2)
    return py

def predict_trajectory_behavior(model,n,optimization=False):
    test = iter(Test)
    x,y =  next(test)
    x,y = x.to(device),y.to(device)
    print(x.shape)
    print(y.shape)
    yl = x[:,:,-1].view(-1,1)
    x = compute_label(x)
    pred = x[:,:,-1].view(-1,1)
    equal = yl.squeeze()==pred.squeeze()
    accu = torch.mean(equal.type(torch.FloatTensor)).item()
    x,y = x.to(device),y.to(device)
    yl = x[:,:,-1].view(-1,1)
    x = compute_label(x)

    pred = x[:,:,-1].view(-1,1)
    equal = yl.squeeze()==pred.squeeze()
    accu = torch.mean(equal.type(torch.FloatTensor)).item()

    # Make predictions on input data x
    predY = model(x)
    criterion = nn.MSELoss()
    test_loss = criterion(predY,y)
    std = WholeSet.std.repeat(x.shape[0],x.shape[1],1)
    std = std[:,:,:4].to(device)
    mn = WholeSet.mn.repeat(x.shape[0],x.shape[1],1)
    mn = mn[:,:,:4].to(device)
    rg = WholeSet.range.repeat(x.shape[0],x.shape[1],1)
    rg = rg[:,:,:4].to(device)
    predY = (predY*(rg*std)+mn).detach().cpu()
    pY = np.array(predY)
    local_labels = (y*(rg*std)+mn).detach().cpu()
    Y = np.array(local_labels)
    pY[:,:-predict_length,:] = Y[:,:-predict_length,:]
    rst_xy = calcu_XY(pY)
    real_predict = torch.from_numpy(Y[:n, - 2:- 1, :4])
    pre_predict = torch.from_numpy(pY[:n, - 2:- 1, :4])
    real_predict1 = torch.from_numpy(Y[:n, - 2:- 1, 2:4])
    rst_predict = torch.from_numpy(rst_xy[:n, - 2:- 1, :2])
    MSE_pre = criterion(real_predict, pre_predict)
    MSE_rst = criterion(real_predict1, rst_predict)
    print("RMSE pre:", math.sqrt(MSE_pre.item()))
    print("RMSE RST:", math.sqrt(MSE_rst.item()))

if __name__ == '__main__':
    # Convert the Training_generator object into an iterator object
    train_iter = iter(Training_generator)
    x, y= next(train_iter)
    print(x.shape,y.shape)
    hidden_size = 256
    # Create an instance of the NNPred class Prednet
    attention_Prednet = NNPred(x.shape[2], y.shape[2],hidden_size, x.shape[0])

    TRAN_TAG = True
    if TRAN_TAG:
        if path.exists("model/attentionStatePredict.pt"):
            attention_Prednet.load_state_dict(torch.load('model/attentionStatePredict.pt'))
        Prednet = attention_Prednet.double()
        Prednet = Prednet.to(device)
        trainIters(Prednet, 200, 0.001, 20)

    # Prednet.load_state_dict(torch.load('model/attentionStatePredict.pt'))
    # Prednet = Prednet.double()
    # Prednet = Prednet.to(device)
    # Prednet.eval()
    # Eval_net(Prednet, True)
    #
    # Prednet.load_state_dict(torch.load('model/attentionStatePredict.pt'))
    # Prednet = Prednet.double()
    # Prednet = Prednet.to(device)
    # Prednet.eval()
    # predict(Prednet, Test, 1, True)
    #
    # Eval_net_behavior(Prednet, True)
    #
    # predict_trajectory_behavior(Prednet, 1, True)

