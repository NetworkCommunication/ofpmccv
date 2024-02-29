from io import open
import os.path
from os import path
import random
import pickle
import pandas as pd
import scipy.signal
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

def load_data(csv_file):
    dataS = pd.read_csv(csv_file)
    F, L, LO, R, RO = 0, 0, 0, 0, 0
    for i in range(len(dataS['Label'].values)):
        if dataS['Label'].values[i] == 'Keep':
            F += 1
            dataS['Label'].values[i] = 0
        if dataS['Label'].values[i] == 'Left':
            L += 1
            dataS['Label'].values[i] = 1
        if dataS['Label'].values[i] == 'Right':
            LO += 1
            dataS['Label'].values[i] = 2
    print('Keep:', F, "Left:", L, "Right:")
    # all-zero array traj
    traj = np.zeros((np.shape(np.asarray(dataS))[0], 68 + 156))
    traj[:, :68] = np.asarray(dataS)
    # Loop iterates each time point k and assigns the time in dataS to time
    for k in range(len(dataS)):
        time = dataS[['Global_Time']]
        time = np.asarray(time)[k]
        frame_time = dataS[dataS.Global_Time == time[0]]
        frame_time = np.asarray(frame_time)
        print(time[0], 'have', len(frame_time), 'vehicles')
        # Calculate the distance between each target vehicle and other vehicles in frame_time
        if frame_time.size > 1:
            dx = np.zeros(np.shape(frame_time)[0])
            dy = np.zeros(np.shape(frame_time)[0])
            vid = np.zeros(np.shape(frame_time)[0])

            for l in range(np.shape(frame_time)[0]):
                dx[l] = frame_time[l][4] - traj[k][4]
                dy[l] = frame_time[l][5] - traj[k][5]
                vid[l] = frame_time[l][0]
            dist = dx * dx + dy * dy
            dist = np.sqrt(dist)

            # # Limit the maximum number of surrounding vehicles
            lim = 39

            if len(dist) > lim:
                idx = np.argsort(dist)
                dx = np.array([dx[i] for i in idx[:lim]])
                dy = np.array([dy[i] for i in idx[:lim]])
                dist = np.array([dist[i] for i in idx[:lim]])
                vid = np.array([vid[i] for i in idx[:lim]])
            xl = dx[dx < -10]
            yl = dy[dx < -10]
            yl = yl[xl>-50]
            distl = dist[dx < -10]
            distl = distl[xl>-50]
            vidl = vid[dx < -10]
            vidl = vidl[xl>-50]
            xl = xl[xl>-50]

            # left top
            yl_top = yl[yl > 0]
            xl_top = xl[yl>0]
            xl_top = xl_top[yl_top<50]
            distl_top = distl[yl>0]
            distl_top = distl_top[yl_top<50]
            vidl_top = vidl[yl>0]
            vidl_top = vidl_top[yl_top<50]
            yl_top = yl_top[yl_top <50]

            # left bot
            yl_bot = yl[ yl < 0]
            xl_bot = xl[yl < 0]
            xl_bot = xl_bot[yl_bot>-50]
            distl_bot = distl[yl < 0]
            distl_bot = distl_bot[yl_bot>-50]
            vidl_bot = vidl[yl < 0]
            vidl_bot = vidl_bot[yl_bot>-50]
            yl_bot = yl_bot[yl_bot>-50]

            # center
            xc = dx[dx >= -10]
            yc = dy[dx >= -10]
            distc = dist[dx >= -10]
            vidc = vid[dx >= -10]

            yc = yc[xc < 10]
            distc = distc[xc < 10]
            vidc = vidc[xc < 10]
            xc = xc[xc < 10]

            # center top
            yc_top = yc[yc > 0]
            xc_top = xc[yc > 0]
            xc_top = xc_top[yc_top < 50]
            distc_top = distc[yc > 0]
            distc_top = distc_top[yc_top < 50]
            vidc_top = vidc[yc > 0]
            vidc_top = vidc_top[yc_top < 50]
            yc_top = yc_top[yc_top < 50]

            # center bot
            yc_bot = yc[yc < 0]
            xc_bot = xc[yc < 0]
            xc_bot = xc_bot[yc_bot > -50]
            distc_bot = distc[yc < 0]
            distc_bot = distc_bot[yc_bot > -50]
            vidc_bot = vidc[yc < 0]
            vidc_bot = vidc_bot[yc_bot > -50]
            yc_bot = yc_bot[yc_bot > -50]

            # right
            xr = dx[dx < 50]
            yr = dy[dx < 50]
            yr = yr[xr > 10]
            distr = dist[dx < 50]
            distr = distr[xr > 10]
            vidr = vid[dx < 50]
            vidr = vidr[xr > 10]
            xr = xr[xr > 10]

            # reft top
            yr_top = yr[yr > 0]
            xr_top = xr[yr > 0]
            xr_top = xr_top[yr_top < 50]
            distr_top = distr[yr > 0]
            distr_top = distr_top[yr_top < 50]
            vidr_top = vidr[yr > 0]
            vidr_top = vidr_top[yr_top < 50]
            yr_top = yr_top[yr_top < 50]

            # reft bot
            yr_bot = yr[yr < 0]
            xr_bot = xr[yr < 0]
            xr_bot = xr_bot[yr_bot > -50]
            distr_bot = distr[yr < 0]
            distr_bot = distr_bot[yr_bot > -50]
            vidr_bot = vidr[yr < 0]
            vidr_bot = vidr_bot[yr_bot > -50]
            yr_bot = yr_bot[yr_bot > -50]

            mini_top = 7
            mini_bot = 6

            # left top
            iy = np.argsort(distl_top)
            iy = iy[0:min(mini_top, len(distl_top))]
            ltop = len(iy)
            xl_top = np.array([xl_top[i] for i in iy])
            yl_top = np.array([yl_top[i] for i in iy])
            distl_top = np.array([distl_top[i] for i in iy])
            vidl_top = np.array([vidl_top[i] for i in iy])

            # left bottom
            iy = np.argsort(distl_bot)
            iy = iy[0:min(mini_bot, len(distl_bot))]
            lbot = len(iy)
            xl_bot = np.array([xl_bot[i] for i in iy])
            yl_bot = np.array([yl_bot[i] for i in iy])
            distl_bot = np.array([distl_bot[i] for i in iy])
            vidl_bot = np.array([vidl_bot[i] for i in iy])

            iy = np.argsort(distc_top)
            iy = iy[0:min(mini_top, len(distc_top))]
            ctop = len(iy)
            xc_top = np.array([xc_top[i] for i in iy])
            yc_top = np.array([yc_top[i] for i in iy])
            distc_top = np.array([distc_top[i] for i in iy])
            vidc_top = np.array([vidc_top[i] for i in iy])

            # center top
            iy = np.argsort(distc_bot)
            iy = iy[0:min(mini_bot, len(distc_bot))]
            cbot = len(iy)
            xc_bot = np.array([xc_bot[i] for i in iy])
            yc_bot = np.array([yc_bot[i] for i in iy])
            distc_bot = np.array([distc_bot[i] for i in iy])
            vidc_bot = np.array([vidc_bot[i] for i in iy])

            # center bottom
            iy = np.argsort(distr_top)
            iy = iy[0:min(mini_top, len(distr_top))]
            rtop = len(iy)
            xr_top = np.array([xr_top[i] for i in iy])
            yr_top = np.array([yr_top[i] for i in iy])
            distr_top = np.array([distr_top[i] for i in iy])
            vidr_top = np.array([vidr_top[i] for i in iy])

            # right top
            iy = np.argsort(distr_bot)
            iy = iy[0:min(mini_bot, len(distr_bot))]
            rbot = len(iy)
            xr_bot = np.array([xr_bot[i] for i in iy])
            yr_bot = np.array([yr_bot[i] for i in iy])
            distr_bot = np.array([distr_bot[i] for i in iy])
            vidr_bot = np.array([vidr_bot[i] for i in iy])

            # left bot
            for i in range(lbot):
                traj[k, 68 + i * 4] = vidl_bot[i]
                traj[k, 68 + 1 + i * 4] = xl_bot[i]
                traj[k, 68 + 2 + i * 4] = yl_bot[i]
                traj[k, 68 + 3 + i * 4] = distl_bot[i]
            if lbot < mini_bot:
                for i in range(mini_bot - lbot):
                    traj[k, 68 + 3 + (lbot - 1) * 4 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (lbot - 1) * 4 + 1 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (lbot - 1) * 4 + 1 + 2 + i * 4] = 0
                    traj[k, 68 + 3 + (lbot - 1) * 4 + 1 + 3 + i * 4] = 0
                # left top
            for i in range(ltop):
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + i * 4] = vidl_top[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 1 + i * 4] = xl_top[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 2 + i * 4] = yl_top[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + i * 4] = distl_top[i]
            if ltop < mini_top:
                for i in range(mini_top - ltop):
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (ltop - 1) * 4 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (ltop - 1) * 4 + 1 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (ltop - 1) * 4 + 1 + 2 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (ltop - 1) * 4 + 1 + 3 + i * 4] = 0
            # center bot
            for i in range(cbot):
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + i * 4] = vidc_bot[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 1 + i * 4] = xc_bot[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 2 + i * 4] = yc_bot[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + i * 4] = distc_bot[i]
            if cbot < mini_bot:
                for i in range(mini_bot - cbot):
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (cbot - 1) * 4 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (   cbot - 1) * 4 + 1 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (cbot - 1) * 4 + 1 + 2 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( cbot - 1) * 4 + 1 + 3 + i * 4] = 0
            # center top
            for i in range(ctop):
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + i * 4] = vidc_top[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( mini_bot - 1) * 4 + 1 + 1 + i * 4] = xc_top[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 2 + i * 4] = yc_top[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( mini_bot - 1) * 4 + 1 + 3 + i * 4] = distc_top[i]
            if ctop < mini_top:
                for i in range(mini_top - ctop):
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( mini_bot - 1) * 4 + 1 + 3 + (ctop - 1) * 4 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( mini_bot - 1) * 4 + 1 + 3 + (ctop - 1) * 4 + 1 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (ctop - 1) * 4 + 1 + 2 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (ctop - 1) * 4 + 1 + 3 + i * 4] = 0
            # right bot
            for i in range(rbot):
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + ( mini_top - 1) * 4 + 1 + i * 4] = vidr_bot[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 1 + i * 4] = xr_bot[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 2 + i * 4] = yr_bot[i]
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + i * 4] = distr_bot[i]
            if rbot < mini_bot:
                for i in range(mini_bot - rbot):
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (rbot - 1) * 4 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (rbot - 1) * 4 + 1 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (rbot - 1) * 4 + 1 + 2 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( rbot - 1) * 4 + 1 + 3 + i * 4] = 0
            # right top
            for i in range(rtop):
                traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + i * 4] = vidr_top[i]
                traj[ k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + ( mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 1 + i * 4] = xr_top[i]
                traj[ k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 2 + i * 4] = yr_top[i]
                traj[ k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + i * 4] = distr_top[i]
            if rtop < mini_top:
                for i in range(mini_top - rtop):
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (rtop - 1) * 4 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + ( rtop - 1) * 4 + 1 + 1 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + ( mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (rtop - 1) * 4 + 1 + 2 + i * 4] = 0
                    traj[k, 68 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (mini_top - 1) * 4 + 1 + 3 + (mini_bot - 1) * 4 + 1 + 3 + (rtop - 1) * 4 + 1 + 3 + i * 4] = 0
    columns = ['Vehicle_ID', 'Frame_ID', 'Total_Frames', 'Global_Time', 'Local_X', 'Local_Y', 'Global_X', 'Global_Y',
               'v_Length', 'v_Width', 'v_Class', 'v_Vel', 'v_Acc', 'Lane_ID', 'Preceeding', 'Following', 'Space_Hdwy', 'Time_Hdwy',
               'Angle', 'L_rX', 'L_rY','L_rVel', 'L_rAcc', 'L_angle', 'L_Class', 'L_Len', 'L_Width', 'F_rX', 'F_rY', 'F_rVel', 'F_rAcc',
               'F_angle', 'F_Class', 'F_Len', 'F_Width','LL_rX', 'LL_rY', 'LL_rVel', 'LL_rAcc', 'LL_angle', 'LL_Class', 'LL_Len', 'LL_Width', 'LF_rX', 'LF_rY',
               'LF_rVel', 'LF_rAcc', 'LF_angle', 'LF_Class', 'LF_Len', 'LF_Width', 'RL_rX', 'RL_rY', 'RL_rVel', 'RL_rAcc', 'RL_angle', 'RL_Class',
               'RL_Len', 'RL_Width', 'RF_rX', 'RF_rY','RF_rVel', 'RF_rAcc', 'RF_angle', 'RF_Class', 'RF_Len', 'RF_Width', 'Label']
    print(len(columns))
    left_bot = []
    for i in range(1, mini_bot + 1):
        vidl_bot_ = 'Vidl_bot_{}'.format(i)
        left_bot.append(vidl_bot_)
        xl_bot_ = 'Xl_bot_{}'.format(i)
        left_bot.append(xl_bot_)
        yl_bot_ = 'Yl_bot_{}'.format(i)
        left_bot.append(yl_bot_)
        distl_bot_ = 'Distl_bot_{}'.format(i)
        left_bot.append(distl_bot_)
    left_top = []
    for i in range(1, mini_top + 1):
        vidl_top_ = 'Vidl_top_{}'.format(i)
        left_top.append(vidl_top_)
        xl_top_ = 'Xl_top_{}'.format(i)
        left_top.append(xl_top_)
        yl_top_ = 'Yl_top_{}'.format(i)
        left_top.append(yl_top_)
        distl_top_ = 'Distl_top_{}'.format(i)
        left_top.append(distl_top_)
    center_bot = []
    for i in range(1, mini_bot + 1):
        vidc_bot_ = 'Vidc_bot_{}'.format(i)
        center_bot.append(vidc_bot_)
        xc_bot_ = 'Xc_bot_{}'.format(i)
        center_bot.append(xc_bot_)
        yc_bot_ = 'Yc_bot_{}'.format(i)
        center_bot.append(yc_bot_)
        distc_bot_ = 'Distc_bot_{}'.format(i)
        center_bot.append(distc_bot_)
    center_top = []
    for i in range(1, mini_top + 1):
        vidc_top_ = 'Vidc_top_{}'.format(i)
        center_top.append(vidc_top_)
        xc_top_ = 'Xc_top_{}'.format(i)
        center_top.append(xc_top_)
        yc_top_ = 'Yc_top_{}'.format(i)
        center_top.append(yc_top_)
        distc_top_ = 'Distc_top_{}'.format(i)
        center_top.append(distc_top_)
    right_bot = []
    for i in range(1, mini_bot + 1):
        vidr_bot_ = 'Vidr_bot_{}'.format(i)
        right_bot.append(vidr_bot_)
        xr_bot_ = 'Xr_bot_{}'.format(i)
        right_bot.append(xr_bot_)
        yr_bot_ = 'Yr_bot_{}'.format(i)
        right_bot.append(yr_bot_)
        distr_bot_ = 'Distr_bot_{}'.format(i)
        right_bot.append(distr_bot_)
    right_top = []
    for i in range(1, mini_top + 1):
        vidr_top_ = 'Vidr_top_{}'.format(i)
        right_top.append(vidr_top_)
        xr_top_ = 'Xr_top_{}'.format(i)
        right_top.append(xr_top_)
        yr_top_ = 'Yr_top_{}'.format(i)
        right_top.append(yr_top_)
        distr_top_ = 'Distr_top_{}'.format(i)
        right_top.append(distr_top_)

    columns = columns + left_bot + left_top + center_bot + center_top + right_bot + right_top
    print(len(columns))
    pd_data = pd.DataFrame(traj, columns=columns)
    print(left_bot,left_top,center_bot,center_top,right_bot,right_top)
    pd_data.to_csv('./data/interactive_formation_improve.csv')

if __name__ == '__main__':
    load_data('./data/my_dataset.csv')