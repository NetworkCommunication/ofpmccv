import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from numpy import mean
from torch import nn

from trajectory_prediction.behavior_model import LSTM
from trajectory_prediction.inputData import get_dataloaderById
from trajectory_prediction.model import NNPred
from trajectory_prediction.trajectory_dataPrepare8 import get_dataloader

torch.cuda.set_device(0)
csvTurn = './my_data/interactive_formation_improve.csv'
length = 50
predict_length = 30
hidden_size = 256
dataS = pd.read_csv(csvTurn)

Training_generator, Test, Valid, WholeSet = get_dataloader(128,length,predict_length)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

# get behavior
def compute_label(input_x, wholeDataSet):
    std = wholeDataSet.std.repeat(input_x.shape[0],input_x.shape[1],1)
    std = std.to(device)
    mn = wholeDataSet.mn.repeat(input_x.shape[0],input_x.shape[1],1)
    mn = mn.to(device)
    rg = wholeDataSet.range.repeat(input_x.shape[0],input_x.shape[1],1)
    rg = rg.to(device)
    input_return = (input_x*(rg*std)+mn).detach().cpu()
    net_behavior = LSTM(1,5,256,2,26)
    net_behavior.load_state_dict(torch.load('model/behavior_prediction.pth'))
    net_behavior.to(device)
    inputs = torch.from_numpy(input_return[:,:,:-1].detach().cpu().numpy().astype(np.float32)).to(device)
    inputs = inputs.view(-1,inputs.shape[2]).unsqueeze(2)
    h = net_behavior.init_hidden(inputs.shape[0])
    h = tuple([each.data for each in h])
    behavior,h = net_behavior(inputs,h)
    _, class_ = torch.max(behavior, dim=1)
    class_ = torch.from_numpy(class_.detach().cpu().numpy().astype(np.double)).to(device)
    class_ = class_.view(input_x.shape[0],input_x.shape[1],-1)
    y_ = input_return[:,:,-1].view(-1,1).numpy().tolist()
    std = std[:,:,-1].unsqueeze(2)
    rg = rg[:,:,-1].unsqueeze(2)
    mn = mn[:,:,-1].unsqueeze(2)
    class_ = (class_ -mn) / (std*rg)
    py = torch.cat((input_x[:,:,:-1],class_),2)
    return py

# Combine behavior with predictions
def predict_trajectory_behavior(model, selectID, n, currentTime, optimization=False):
    DatasetById, wholeDataSet = get_dataloaderById(128, length, predict_length, selectID, currentTime)
    test = iter(DatasetById)
    x,y = next(test)
    x,y = x.to(device),y.to(device)
    print(x.shape)
    print(y.shape)
    x = compute_label(x, wholeDataSet)

    print(x.shape)
    pred = x[:, :, -1].view(-1, 1)
    predY = model(x)
    criterion = nn.MSELoss()
    test_loss = criterion(predY,y)
    std = wholeDataSet.std.repeat(x.shape[0],x.shape[1],1)
    std = std[:,:,:4].to(device)
    mn = wholeDataSet.mn.repeat(x.shape[0],x.shape[1],1)
    mn = mn[:,:,:4].to(device)
    rg = wholeDataSet.range.repeat(x.shape[0],x.shape[1],1)
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
    result_dict = {}
    if MSE_rst.item() < 1000:
        fig = plt.figure()
        plt.plot(Y[:n,:-predict_length,2][0],Y[:n,:-predict_length,3][0],'r',label = 'history state')
        if optimization:
            plt.plot(rst_xy[:n,-predict_length-1:,0][0],rst_xy[:n,-predict_length-1:,1][0],'b',label="prediction state")
            print("Vehicle",selectID, ":x,y:", rst_xy[:n,-predict_length-1:,0][0][-1],rst_xy[:n,-predict_length-1:,1][0][-1])
            print("Vehicle",selectID, ":v,a：", Y[:n,predict_length:,1][0][-1],Y[:n,predict_length:,0][0][-1])

        print("Test Loss:",test_loss.item())
        plt.close(fig)

        result_dict[selectID] = {
            "RMSE": RMSE_rst.item(),
            "x": rst_xy[:n, -predict_length - 1:, 0][0][-1],
            "y": rst_xy[:n, -predict_length - 1:, 1][0][-1],
            "v": Y[:n, predict_length:, 1][0][-1],
            "a": Y[:n, predict_length:, 0][0][-1]
        }
        return result_dict

# Get turn flag
def get_turn_signal():
    turn_data = dataS[dataS['Turn'] == 1]
    global_time = turn_data.iloc[0]['Global_Time']
    global_time = turn_data.iloc[0]['Global_Time']


    return turn_data, global_time

# Get vehicle data
def get_vehicle_data(turn_data):

    vehicle_ids = turn_data['Vehicle_ID'].values
    selected_data = dataS[dataS['Vehicle_ID'].isin(vehicle_ids)]


    myData = []
    for i in range(len(selected_data)):
        for col_idx in range(69, len(turn_data.columns) - 1, 4):
            column_name = turn_data.columns[col_idx]
            myData1 = selected_data.loc[selected_data[column_name] != 0, column_name]
            if myData1.tolist() not in [d.tolist() for d in myData]:
                myData.append(myData1)

    myData = list(set([tuple(i) for i in myData]))
    data = list(set([tuple(set(item)) for item in myData if item]))
    result = list(set().union(*data))
    print(result)
    return result

# Get IDs of surrounding vehicles
def get_surrounding_vehicles(turn_data, sourround_vehicle_id):
    vehicle_ids = turn_data['Vehicle_ID'].values
    selected_data = dataS[dataS['Vehicle_ID'].isin(vehicle_ids)]

    resultSurID = {}
    for element in sourround_vehicle_id:
        resultSurID[element] = dataS[dataS['Vehicle_ID'] == element]
    return resultSurID

# Processing trajectory data
def process_trajectory(CurrentTime, surrounding_vehicle_data):

    new_dict = {}
    currentTime = int(CurrentTime)
    for key, value in surrounding_vehicle_data.copy().items():
        # print(value)
        new_value = []
        if len(value[value['Global_Time'] == currentTime].index) > 0:
            row_num = value[value['Global_Time'] == currentTime].index[0]
            data = value.iloc[:row_num - value.iloc[0, 0], :]
            new_value.append(data)
        else:
            continue

        new_dict[key] = new_value

    return new_dict

# get Result
def getResultXY(trajectoryDataSet, global_time):
    Prednet = NNPred(26, 4, hidden_size, 128)
    Prednet.load_state_dict(torch.load('model/trajectory_predict.pt'))
    Prednet = Prednet.double()
    Prednet = Prednet.to(device)

    result_dict = {}
    for key, value in trajectoryDataSet.copy().items():
        print("*" * 6, key)
        result_dict[key] = predict_trajectory_behavior(Prednet, key, 1, global_time, True)

    result_dict = {k: v for k, v in result_dict.items() if v is not None}
    return result_dict

if __name__ == "__main__":
    turn_signal, global_time = get_turn_signal()

    vehicle_data = get_vehicle_data(turn_signal)

    surrounding_vehicle_data = get_surrounding_vehicles(turn_signal, vehicle_data)

    trajectory_process = process_trajectory(global_time, surrounding_vehicle_data)

    result = getResultXY(trajectory_process, global_time)

    rmse_values = [v[key]['RMSE'] for k, v in result.items() for key in v]

    avg_rmse = mean(rmse_values)
    print("avg_RMSE:", avg_rmse)

    print(result)
