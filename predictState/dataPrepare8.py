from io import open
import random
from os import path

import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np

class TrajectoryDataset(Dataset):

    def __init__(self, length=60, predict_length=30, csv_file='./dataFilling/data/deleteDataFillAlg.csv'):
        """
            Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
            on a sample.
            """
        self.csv_file = csv_file
        self.length = length
        self.predict_length = predict_length
        self.X_frames_trajectory = []
        self.Y_frames_trajectory = []
        self.load_data()
        self.normalize_data()

    # Amount of vehicle trajectory data
    def __len__(self):
        return len(self.X_frames_trajectory)

    def __getitem__(self,idx):
        single_trajectory_data = self.X_frames_trajectory[idx]
        single_trajectory_label = self.Y_frames_trajectory[idx]
        return (single_trajectory_data, single_trajectory_label)

    def load_data(self):
        dataS = pd.read_csv(self.csv_file)

        for i in range(len(dataS['Angle'].values)):
            dataS["Angle"].values[i] = dataS["Angle"].values[i] * 180 / np.pi
        max_vehiclenum = np.max(dataS.Vehicle_ID.unique())
        count_ = []

        # It is guaranteed to be the trajectory of the same vehicle, and the data is loaded one by one.
        for vid in dataS.Vehicle_ID.unique():
            frame_ori = dataS[dataS.Vehicle_ID == vid]
            # Only keep the following columns
            frame = frame_ori[['Local_X', 'Local_Y', 'v_Vel', 'v_Acc', 'Angle',
                                'Xl_bot_1', 'Yl_bot_1', 'Distl_bot_1','Xl_top_1', 'Yl_top_1', 'Distl_top_1', 'Xc_bot_1',
                               'Yc_bot_1', 'Distc_bot_1',  'Xc_top_1', 'Yc_top_1', 'Distc_top_1',   'Xr_bot_1', 'Yr_bot_1', 'Distr_bot_1',
                               'Xr_top_1', 'Yr_top_1', 'Distr_top_1','Label']]
            frame = np.asarray(frame)
            frame[np.where(frame > 4000)] = 0
            dis = frame[1:, :2] - frame[:-1, :2]
            dis = dis.astype(np.float64)
            dis = np.sqrt(np.power(dis[:, 0], 2) + np.power(dis[:, 1], 2))
            idx = np.where(dis > 10)
            if not (idx[0].all):
                print("discontinuious trajectory")
                continue
            frame[:, 0:2] = scipy.signal.savgol_filter(frame[:, 0:2], window_length=51, polyorder=3, axis=0)

            # calculate vel_x and vel_y according to local_x and local_y for all vehi
            All_vels = []
            for i in range(1):
                x_vel = (frame[1:, 0 + i * 5] - frame[:-1, 0 + i * 5]) / 0.1;
                v_avg = (x_vel[1:] + x_vel[:-1]) / 2.0;
                v_begin = [2.0 * x_vel[0] - v_avg[0]];
                v_end = [2.0 * x_vel[-1] - v_avg[-1]];
                velx = (v_begin + v_avg.tolist() + v_end)
                velx = np.array(velx)

                y_vel = (frame[1:, 1 + i * 5] - frame[:-1, 1 + i * 5]) / 0.1;
                vy_avg = (y_vel[1:] + y_vel[:-1]) / 2.0;
                vy1 = [2.0 * y_vel[0] - vy_avg[0]];
                vy_end = [2.0 * y_vel[-1] - vy_avg[-1]];
                vely = (vy1 + vy_avg.tolist() + vy_end)
                vely = np.array(vely)

                if isinstance(All_vels, (list)):
                    All_vels = np.vstack((velx, vely))
                else:
                    All_vels = np.vstack((All_vels, velx.reshape(1, -1)))
                    All_vels = np.vstack((All_vels, vely.reshape(1, -1)))
            All_vels = np.transpose(All_vels)
            total_frame_data = np.concatenate((All_vels[:, :2], frame), axis=1)
            if (total_frame_data.shape[0] < 364):
                continue
            X = total_frame_data[:-self.predict_length, :]
            Y = total_frame_data[self.predict_length:, :4]

            count = 0
            # This loop is to divide the training data into many sequences of length self.length for training the model
            for i in range(X.shape[0] - self.length):
                if random.random() > 0.2:
                    continue
                j = i - 1;
                if count > 60:
                    break
                self.X_frames_trajectory = self.X_frames_trajectory + [
                    X[i:i + self.length, :]]
                self.Y_frames_trajectory = self.Y_frames_trajectory + [Y[i:i + self.length, :]]
                count = count + 1
            count_.append(count)

    # standardization
    def normalize_data(self):
        A = [list(x) for x in zip(*(self.X_frames_trajectory))]
        A = np.array(A).astype(np.float64)
        A = torch.from_numpy(A)
        print(A.shape)
        A = A.view(-1, A.shape[2])
        print('A:', A.shape)


        self.mn = torch.mean(A, dim=0)
        self.range = (torch.max(A, dim=0).values - torch.min(A, dim=0).values) / 2.0
        self.range = torch.ones(self.range.shape, dtype=torch.double)
        self.std = torch.std(A, dim=0)
        print(self.std[-1])

        self.X_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn) / (self.std * self.range) for item in
            self.X_frames_trajectory]
        self.Y_frames_trajectory = [
            (torch.from_numpy(np.array(item).astype(np.float64)) - self.mn[:4]) / (self.std[:4] * self.range[:4]) for
            item in self.Y_frames_trajectory]

def get_dataloader(BatchSize=64, length=60, predict_length=30):
    '''
    return torch.util.data.Dataloader for train,test and validation
    '''
    # load dataset
    if path.exists("pickle/process_traj_0903_{}_{}.pickle".format(predict_length, length)):
        with open('pickle/process_traj_0903_{}_{}.pickle'.format(predict_length, length), 'rb') as data:
            dataset = pickle.load(data)

    else:
        dataset = TrajectoryDataset(length, predict_length)
        with open('pickle/process_traj_0903_{}_{}.pickle'.format(predict_length, length), 'wb') as output:
            pickle.dump(dataset, output)
    legth_traj = dataset.__len__()
    num_train_traj = (int)(legth_traj * 0.8)
    num_test_traj = (int)(legth_traj * 0.9) - num_train_traj
    num_validation_traj = (int)(legth_traj - num_test_traj - num_train_traj)

    train_traj, test_traj, validation_traj = torch.utils.data.random_split(dataset, [num_train_traj, num_test_traj, num_validation_traj])

    train_loader_traj = DataLoader(train_traj, batch_size=BatchSize, shuffle=True)
    test_loader_traj = DataLoader(test_traj, batch_size=BatchSize, shuffle=True)
    validation_loader_traj = DataLoader(validation_traj, batch_size=BatchSize, shuffle=True)
    iters = iter(train_loader_traj)
    x_trajectory, y_trajectory = next(iters)
    return (train_loader_traj, test_loader_traj, validation_loader_traj, dataset)

if __name__ == '__main__':
    get_dataloader()