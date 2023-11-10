# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
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
from torch.autograd import Variable
import torch.distributions as dist
from dataPrepare8 import *
import torch
import numpy as np

# torch.manual_seed(0)

MAX_LENGTH = 100
torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print('ok')
length = 90
predict_length = 50
Training_generator, Test, Valid, WholeSet = get_dataloader(128, length, predict_length)
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

        weights = torch.matmul(qs, ks.permute(0, 1, 3, 2))
        weights = weights / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        weights = F.softmax(weights, dim=-1)

        attention = torch.matmul(weights, vs)

        reshaped = attention.view(batch_size, seq_length, self.num_heads * self.head_dim)
        output = self.fc_layer(reshaped)

        return output

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

        out = mlp_out + self.in2out(input)

        out, _ = self.decoder_lstm(out)

        output = self.fc2(out)

        return output


def trainIters(encoder, epoches, learning_rate=0.0001, print_every=7):

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, mode='min', factor=0.1,
                                                           patience=120, verbose=True, threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-015)
    criterion = nn.MSELoss()
    loss_min = np.inf
    train_losses = []
    valid_losses = []
    encoder.train()
    for e in range(1, epoches + 1):
        train_loss = []
        for batch_i, (local_batch, local_labels) in enumerate(Training_generator):
            encoder.zero_grad()
            local_batch = local_batch[:, :, :-1].to(device)
            local_labels = local_labels.to(device)

            predY = encoder(local_batch)
            loss = criterion(predY, local_labels).to(device)
            loss.backward()
            encoder_optimizer.step()
            train_loss.append(loss.item())
            if batch_i % print_every == 0:
                valid_loss = []
                encoder.eval()
                for x, y in Valid:
                    x, y = x[:, :, :-1].to(device), y.to(device)
                    predict = encoder(x)
                    loss_valid = criterion(predict, y)
                    valid_loss.append(loss_valid.item())
                encoder.train()
                train_loss_mean = np.mean(train_loss)
                valid_loss_mean = np.mean(valid_loss)
                train_losses.append(train_loss_mean)
                valid_losses.append(valid_loss_mean)
                scheduler.step(valid_loss_mean)
                print("Epoch:{}/{},Step:{}/{}".format(e, epoches, batch_i, len(Training_generator)),
                      "Train_Loss:{},Valid_Loss: {}".format(train_loss_mean, valid_loss_mean))
                if valid_loss_mean < loss_min:
                    print("valid_loss decrease!!!save the model.")
                    loss_min = valid_loss_mean
                    torch.save(encoder.state_dict(), 'model/atten_statePredict_nobehave.pt')


def predict(model, test_load, n, optimizer=False):
    test = iter(test_load)
    x, y = next(test)
    x, y = x.to(device), y.to(device)
    predY = model(x[:, :, :-1])
    criterion = nn.MSELoss()
    test_loss = criterion(predY, y)
    std = WholeSet.std.repeat(x.shape[0], x.shape[1], 1)
    std = std[:, :, :4].to(device)
    mn = WholeSet.mn.repeat(x.shape[0], x.shape[1], 1)
    mn = mn[:, :, :4].to(device)
    rg = WholeSet.range.repeat(x.shape[0], x.shape[1], 1)
    rg = rg[:, :, :4].to(device)
    predY = (predY * (rg * std) + mn).detach().cpu()
    pY = np.array(predY)
    pY = scipy.signal.savgol_filter(pY, window_length=((x.shape[1] - 1) // 2) * 2 + 1, polyorder=3, axis=1)
    local_labels = (y * (rg * std) + mn).detach().cpu()
    Y = np.array(local_labels)
    pY[:, :-predict_length, :] = Y[:, :-predict_length, :]

    real_predict = torch.from_numpy(Y[:n, - 2:- 1, :4])
    pre_predict = torch.from_numpy(pY[:n, - 2:- 1, :4])
    MSE_pre = criterion(real_predict, pre_predict)
    print("RMSE pre:", math.sqrt(MSE_pre.item()))
    print("Test Loss:", test_loss.item())

if __name__ == '__main__':
    train_iter = iter(Training_generator)
    x, y = next(train_iter)
    print(x.shape, y.shape)
    hidden_size = 256
    Prednet = NNPred(x.shape[2] - 1, y.shape[2], hidden_size, x.shape[0])

    TRAN_TAG = True
    if TRAN_TAG:
        if path.exists("model/atten_statePredict_nobehave.pt"):
            Prednet.load_state_dict(torch.load('model/atten_statePredict_nobehave.pt'))
        Prednet = Prednet.double()
        Prednet = Prednet.to(device)
        # trainIters(Prednet, 200, 0.001, 120)

    # Prednet.load_state_dict(torch.load('model/trajectory_predict_25_4_30.pt'))
    # Prednet = Prednet.double()
    # Prednet = Prednet.to(device)
    # Prednet.eval()
    # Eval_net(Prednet, True)
    #
    # Prednet.load_state_dict(torch.load('model/trajectory_predict_25_4_30.pt'))
    # Prednet = Prednet.double()
    # Prednet = Prednet.to(device)
    # Prednet.eval()
    predict(Prednet, Test, 1, True)

