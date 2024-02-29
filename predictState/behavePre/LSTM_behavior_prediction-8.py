from time import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

from dataPrepare8 import get_dataloader

torch.cuda.set_device(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network structure for driving behavior prediction
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers):
        super(LSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_dim, n_layers, dropout=0., batch_first=True, bidirectional=False)

        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc = nn.Linear(hidden_dim * 2, output_size)
        self.fc2 = nn.Linear(X_train.shape[2], output_size)

    def forward(self, x, hidden):
        batch_size = x.size(0)

        r_out, hidden = self.lstm(x, hidden)
        """
        r_out = torch.mean(r_out,dim=2).squeeze()

        output= self.fc2(r_out)
        # shape output to be (batch_size*seq_length, hidden_dim)
        """
        r_out = r_out.contiguous().view(-1, self.hidden_dim)

        output = self.fc1(r_out)
        output = self.fc(output)
        output = output.view(batch_size, -1, 5)
        output = output[:, -1]

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


def state_to_behavior_data(x):
    x = x.to(device)
    std = WholeSet.std.repeat(x.shape[0],x.shape[1],1)
    std = std.to(device)
    mn = WholeSet.mn.repeat(x.shape[0],x.shape[1],1)
    mn = mn.to(device)
    rg = WholeSet.range.repeat(x.shape[0],x.shape[1],1)
    rg = rg.to(device)
    input_return = (x*(rg*std)+mn).detach().cpu()
    inputs = torch.from_numpy(input_return[:,:,:-1].detach().cpu().numpy().astype(np.float32))
    inputs = inputs.view(-1,inputs.shape[2])
    y = torch.from_numpy(input_return[:,:,-1].detach().cpu().numpy().astype(int))
    y = y.view(-1,1).squeeze()
    return inputs,y


# Training model
def train(net, epochs, train_loader, valid_loader, clip, lr=0.0002):
    loss_min = np.inf
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    counter = 0
    losses_train = []
    losses_valid = []
    accuracies_e = []
    for e in range(epochs):
        net.train()
        train_loss = []
        for inputs, labels in train_loader:
            inputs, labels = state_to_behavior_data(inputs)
            h = net.init_hidden(inputs.shape[0])
            h = tuple([each.data for each in h])
            inputs = inputs.unsqueeze(2)
            if (train_on_gpu):
                inputs, labels = torch.from_numpy(inputs.numpy().astype(np.float32)).to(device), labels.long().to(
                    device)
            net.zero_grad()
            output, h = net(inputs, h)
            loss = criterion(output, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()
            train_loss.append(loss.item())
        val_losses = []
        net.eval()
        accuracies = []
        for inputs, labels in valid_loader:
            inputs, labels = state_to_behavior_data(inputs)
            val_h = net.init_hidden(inputs.shape[0])
            inputs = inputs.unsqueeze(2)
            val_h = tuple([each.data for each in val_h])
            if (train_on_gpu):
                inputs, labels = torch.from_numpy(inputs.numpy().astype(np.float32)).to(device), labels.long().to(
                    device)
            output, val_h = net(inputs, val_h)
            val_loss = criterion(output, labels)
            _, class_ = torch.max(output, dim=1)
            equal = class_ == labels.view(class_.shape)
            accuracy = torch.mean(equal.type(torch.FloatTensor)).item()
            val_losses.append(val_loss.item())
            accuracies.append(accuracy)
        net.train()
        losses_train.append(np.mean(train_loss))
        losses_valid.append(np.mean(val_losses))
        accuracies_e.append(np.mean(np.mean(accuracies)))
        print("Epoch: {}/{}...".format(e + 1, epochs),
              "Loss: {}...".format(np.mean(train_loss)),
              "Val Loss: {}...".format(np.mean(val_losses)),
              "val accuracy:{}.".format(np.mean(accuracies))
              )
        if np.mean(val_losses) < loss_min:
            print('Val loss decreased...')
            torch.save(net.state_dict(), 'model/behavior_prediction.pth')
            loss_min = np.mean(val_losses)
    print('min loss', loss_min)
    df = pd.DataFrame({'losses_train': losses_train})
    df.to_csv('./result/beh/losses_train.csv', index=False, header=False)
    df = pd.DataFrame({'losses_valid': losses_valid})
    df.to_csv('./result/beh/losses_valid.csv', index=False, header=False)
    return accuracies_e


# test model
def test(net, test_loader):
    lr = 0.001

    criterion = nn.CrossEntropyLoss()
    test_losses = []
    accuracies = []
    net.eval()
    class_correct = np.zeros(3)
    class_total = np.zeros(3)
    classes = ["Keep", "Left_change","Right_change"]
    for inputs, y in test_loader:
        inputs, y = state_to_behavior_data(inputs)
        h = net.init_hidden(inputs.shape[0])
        inputs = inputs.unsqueeze(2)
        h = tuple([each.data for each in h])
        if (train_on_gpu):
            inputs, y = torch.from_numpy(inputs.numpy().astype(np.float32)).to(device), y.long().to(device)
        output, h = net(inputs, h)

        test_loss = criterion(output, y)
        _, class_ = torch.max(output, dim=1)
        equal = class_ == y.view(class_.shape)
        for i in range(y.shape[0]):
            label = y.data[i].item()
            class_correct[label] += equal[i].item()
            class_total[label] += 1
        accuracy = torch.mean(equal.type(torch.FloatTensor)).item()
        test_losses.append(test_loss.item())
        accuracies.append(accuracy)

    print('Test Loss: {:.20f}\n'.format(test_loss.item()))
    for i in range(5):
        if class_total[i] > 0:
            print('Test Accuracy of {}:{:.4f}({}/{})'.format(classes[i], 100 * class_correct[i] / class_total[i],
                                                             int(np.sum(class_correct[i])),
                                                             int(np.sum(class_total[i]))))
        else:
            print('Test Accuracy of {}:N/A(no examples)'.format(classes[i]))
    print('Test Accuracy(Overall):{:.4f} ({}/{})'.format(100 * np.sum(class_correct) / np.sum(class_total),
                                                         int(np.sum(class_correct)),
                                                         int(np.sum(class_total))))
    print("Test loss: {:.10f}".format(np.mean(test_losses)), 'Test Accuracy:{}'.format(np.mean(accuracies)))


if __name__ == '__main__':

    train_loader, test_loader, valid_loader, WholeSet = get_dataloader(1, 60, 30)
    iters = iter(train_loader)
    X_train, y = next(iters)

    # Initialize network parameters
    net = LSTM(1, 3, 256, 2)

    if train_on_gpu:
        net.to(device)
    start = time()
    epochs = 200

    # Calculate the accuracy
    accuracy = train(net, epochs, train_loader, valid_loader, clip=5, lr=0.0001)
    print('Training time is:', time() - start, 's')

    df = pd.DataFrame({'accuracy': accuracy})
    df.to_csv('./result/beh/behaveAccuracy.csv', index=False, header=False)

    net_test = LSTM(1, 3, 256, 2)
    net_test.load_state_dict(torch.load('model/behavior_prediction.pth'))
    if train_on_gpu:
        net_test.cuda()
    test(net_test,test_loader)

