from __future__ import absolute_import
from __future__ import print_function
import logging
import math
import time

import gym
import numpy as np
import pandas as pd
from gym import spaces
import random as rn
import os
import sys
import traci
import traci.constants as tc
import torch.nn.functional as F

# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(
        __file__), '..', '..', '..', '..', "tools"))  # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))  # tutorial in docs
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

# 是否打开GUI界面
gui = False
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')

config_path = "StraightRoad.sumocfg"

class LaneChangePredict(gym.Env):
    # 定义了 gym 环境的 metadata，指定了可视化模式为 human（即显示在屏幕上）
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.minAutoVelocity = 0
        self.maxAutoVelocity = 20

        self.minOtherVehVelocity = 0
        self.maxOtherVehVelocity = 20

        self.minDistanceFrontVeh = 0
        self.maxDistanceFrontVeh = 150

        self.minDistanceRearVeh = 0
        self.maxDistanceRearVeh = 150

        self.minLaneNumber = 0
        self.maxLaneNumber = 2 # 车道数为3，但由于获取车道index时，index从0开始计算

        self.CommRange = 150  # 联合感知范围为150米

        self.delta_t = 0.1    # 0.1秒为一个时隙
        self.AutoCarID = 'Car'  # 目标车辆
        self.PrevSpeed = 0
        self.PrevVehDistance = 0
        self.VehicleIds = 0
        self.traFlowNumber = 0   # 前方范围内车辆数
        self.finaTCC = 0

        self.previous_lane_index = 1

        self.overpassFlag = 0
        self.AutoCarFrontID = 'CarF'  # 超车结束标志：前方车辆
        self.ttc_safe = 3  # 最低的安全TTC的值

        # 离散动作：左变道、保持车道、右变道
        self.action_space_vehicle =[-1, 0, 1]   # 0为不变道，-1为左变道，1为右变道
        self.n_actions = len(self.action_space_vehicle)
        self.n_actions = int(self.n_actions)
        # 连续动作：速度变化
        self.param_velocity = [0, 20]
        self.n_features = 18  # 状态的维度

        # self.actions = np.zeros((int(self.n_actions), 1 + 1))  # 第一个1表示索引，第二个1表示变道动作
        self.actions = np.array([[0, -1], [1, 0], [2, 1]])



    def reset(self):
        self.TotalReward = 0
        self.numberOfLaneChanges = 0  # 车辆变道的次数
        self.numberOfOvertakes = 0  # 完成超车的次数
        self.currentTrackingVehId = 'None'
        self.overpassFlag = 0  # 设置完成标志,即是否达到终止条件

        traci.close()

        sumo_binary = "sumo-gui"  # SUMO的可执行文件路径，如果没有设置环境变量，需要指定完整路径
        sumocfg_file = "StraightRoad.sumocfg"  # SUMO配置文件路径

        # sumo_binary = "sumo"  # SUMO的可执行文件路径，如果没有设置环境变量，需要指定完整路径
        # sumocfg_file = "data/Lane3/StraightRoad.sumocfg"  # SUMO配置文件路径

        # sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--delay", "100", "--scale", "1"]
        sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "100", "--scale", "1"]
        traci.start(sumo_cmd)

        # 重置 SUMO 环境并加载配置文件
        # traci.load(config_path)
        print('Resetting the layout')
        # 执行一次仿真步长，模拟 SUMO 执行初始化
        traci.simulationStep()

        self.VehicleIds = traci.vehicle.getIDList()  # 获取模拟环境中所有车辆 ID 集合

        # 为每辆车都订阅指定的变量
        for veh_id in self.VehicleIds:
            traci.vehicle.subscribe(veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION])

        # 创建一个新的初始状态，以便在下一步仿真步骤中使用
        self.state = self._findstate()
        # traci.simulationStep()
        # 返回当前环境状态作为初始状态
        return np.array(self.state)

    # 接受一个 index 参数，在 actions 矩阵中查找并返回对应行的动作数组
    def find_action(self, index):
        return self.actions[index][1]

    def step(self, action, action_param):
        # print("变道结果：", x,";速度结果：", v_n)
        Vehicle_Params = traci.vehicle.getAllSubscriptionResults()
        # 获取执行前车辆的速度
        self.PrevSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
        # 获取执行前车辆所在车道的纵向位置
        self.PrevVehDistance = Vehicle_Params[self.AutoCarID][tc.VAR_LANEPOSITION]

        # 获取车辆的当前车道索引
        lane_index = traci.vehicle.getLaneIndex(self.AutoCarID)

        if lane_index != self.previous_lane_index:
            self.numberOfLaneChanges += 1

        self.previous_lane_index = lane_index
        traci.simulationStep()

        # 更新状态
        self.state = self._findstate()

        # 结束标志
        self.end = self.is_overtake_complete(self.state)

        # 计算奖励
        reward = self.updateReward(action, self.state)
        speed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]

        return self.state, speed, self.end

    # 接受一个 index 参数，在 actions 矩阵中查找并返回对应行的动作数组
    # def find_action(self, index):
    #     return self.actions[index, :]


    def close(self):
        traci.close()

    # 计算车辆之间的距离
    def _findRearVehDistance(self, vehicleparameters):
        # 二维数组parameters，用于存储每辆车的相关信息
        parameters = [[0 for x in range(5)] for x in range(len(vehicleparameters))]
        i = 0
        d1 = -1
        d2 = -1
        d3 = -1
        d4 = -1
        d5 = -1
        d6 = -1
        v1 = -1
        v2 = -1
        v3 = -1
        v4 = -1
        v5 = -1
        v6 = -1
        # 遍历全部车辆的ID
        for VehID in self.VehicleIds:
            parameters[i][0] = VehID
            parameters[i][1] = vehicleparameters[VehID][tc.VAR_LANEPOSITION]  # X position
            parameters[i][2] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # lane Index
            parameters[i][3] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # v
            parameters[i][4] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]  # a
            i = i + 1

        # 通过 X 方向的坐标值升序排序存储在二维数组 parameters 中的车辆列表
        parameters = sorted(parameters, key=lambda x: x[1])  # Sorted in ascending order based on x distance
        # Find Row with Auto Car
        # 找出目标车辆并将记录其在列表中的位置，以及RowIDAuto 变量用于存储下标，值为目标车辆所在行的位置
        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        # 用于计算汽车周围车辆的状态信息，包括各个方向的车辆距离 d、速度 v 等参数，并更新超车次数
        # if there are no vehicles in front
        if RowIDAuto == len(self.VehicleIds) - 1:
            d1 = -1
            v1 = -1
            d3 = -1
            v3 = -1
            d5 = -1
            v5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
            # Check if an overtake has happend
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            # 当前超车的车辆ID也设置为 None
            self.currentTrackingVehId = 'None'
        else:
            # If vehicle is in the lowest lane（最右侧车道）, then d5,d6,v5,v6 do not exist
            if parameters[RowIDAuto][2] == 0:
                d5 = -1
                v5 = -1
                d6 = -1
                v6 = -1
            # if the vehicle is in the maximum lane index（最左侧车道）, then d3.d4.v3.v4 do not exist
            elif parameters[RowIDAuto][2] == (self.maxLaneNumber - 1):
                d3 = -1
                v3 = -1
                d4 = -1
                v4 = -1
            # find d1 and v1  从当前行向下搜索车辆，以查找前方车辆的状态参数
            index = RowIDAuto + 1
            # 如果存在同一车道上的前方车辆，则计算前方车辆与当前车辆之间的距离 d1和速度 v1
            while index != len(self.VehicleIds):
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d1 = parameters[index][1] - parameters[RowIDAuto][1]
                    v1 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d1 = -1
                v1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 150
            # find d3 and v3  从当前行向下搜索车辆，以查找右侧车道的前方车辆的状态参数
            index = RowIDAuto + 1
            # 如果左侧车道存在前方车辆，则计算其于当前车辆之间的距离 d3 和速度 v3
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d3 = parameters[index][1] - parameters[RowIDAuto][1]
                    v3 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d3 = -1
                v3 = -1
            # find d5 and v5
            index = RowIDAuto + 1
            # 如果右侧车道存在前方车辆，则计算其于当前车辆之间的距离 d5 和速度 v5
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d5 = parameters[index][1] - parameters[RowIDAuto][1]
                    v5 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            # there is no vehicle in front
            if index == len(self.VehicleIds):
                d5 = -1
                v5 = -1
            # find d2 and v2
            index = RowIDAuto - 1
            # 如果存在同一车道上的后方车辆，则计算后方车辆与当前车辆之间的距离 d2 速度 v2
            while index >= 0:
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d2 = parameters[RowIDAuto][1] - parameters[index][1]
                    v2 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            # 如果同一车道上没有后方车辆
            if index < 0:
                d2 = -1
                v2 = -1
            # find d4 and v4
            # 类似地，计算右侧和左侧车道的后方车辆状态参数d4、v4、d6 和 v6
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d4 = parameters[RowIDAuto][1] - parameters[index][1]
                    v4 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d4 = -1
                v4 = -1
            # find d6 and v6
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d6 = parameters[RowIDAuto][1] - parameters[index][1]
                    v6 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d6 = -1
                v6 = -1
            # Find if any overtakes has happend
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            # 将当前正在追踪的前方车辆ID设置为当前车道上的下一辆车辆的ID  这个ID存储在列表 parameters 中的第 RowIDAuto + 1 行第一个元素中，即该车辆的ID。这是一个用于跟踪当前车辆前方的车辆的ID。
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0:  # This means that there is no car behind  没有后方车辆
            RearDist = -1
        else:  # There is a car behind return the distance between them
            RearDist = (parameters[RowIDAuto][1] - parameters[RowIDAuto - 1][
                1])  # 如果当前存在后方的车辆，计算当前车辆和后方车辆之间的距离，即当前车辆的位置减去上一行车道上的车辆的位置
        # Return car in front distance
        if RowIDAuto == len(self.VehicleIds) - 1:  # 没有前方车辆
            FrontDist = -1
            # Save the current front vehicle Features
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
        else:
            FrontDist = (parameters[RowIDAuto + 1][1] - parameters[RowIDAuto][
                1])  # 计算当前车辆和前方车辆之间的距离，即下一行车道上的车辆的位置减去当前车辆的位置，这是计算前方车辆间距的方法
            # Save the current front vehicle Features
            self.CurrFrontVehID = parameters[RowIDAuto + 1][0]
            self.CurrFrontVehDistance = FrontDist
        # return RearDist, FrontDist
        return d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6

    def _findstate(self):
        # 使用getAllSubscriptionResults()方法获取已订阅车辆的状态列表
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        # find d1,v1,d2,v2,d3,v3,d4,v4, d5, v5, d6, v6  调用该函数来查找后方的车辆的距离和速度，并将它们分配给相应的变量。
        d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6 = self._findRearVehDistance(VehicleParameters)
        # 检查前方车辆距离 d1是否小于通信范围，如果在通信范围之外，则将其设置为最大可能距离。如果前方没有车辆，则将其设置为最大距离。
        if ((d1 > self.CommRange)):
            d1 = self.maxDistanceFrontVeh
            v1 = -1
        elif d1 < 0:  # if there is no vehicle ahead in L0
            d1 = self.maxDistanceFrontVeh  # as this can be considered as vehicle is far away
        # 检查前方车速 v1 是否为负数，如果为负数，则将其设置为零。这通常会出现在没有前方车辆或者前方车辆被超车时
        if ((v1 < 0) and (d1 <= self.CommRange)):
            # there is no vehicle ahead in L0 or there is a communication error: # there is no vehicle ahead in L0
            v1 = 0

        # 检查后方车辆距离 d2 是否大于通信范围，如果是，则将其设置为最大可能距离。如果后方没有车辆，则将其设置为零，以避免出现负回报
        if ((d2 > self.CommRange)):
            d2 = self.maxDistanceRearVeh
            v2 = -1
        elif d2 < 0:  # There is no vehicle behind in L0
            d2 = 0  # to avoid negetive reward
        # 检查后方车速 v2 是否为负数，如果为负数，则将其设置为零。这通常会出现在没有后方车辆或者后方车辆被超车时
        if ((v2 < 0) and (d2 <= self.CommRange)):
            # there is no vehicle behind in L0 or there is a communication error
            v2 = 0
        if ((d3 > self.CommRange)):
            d3 = self.maxDistanceFrontVeh
            v3 = -1
        elif d3 < 0: # no vehicle ahead in L1
            d3 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v3 < 0) and (d3 <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v3 = 0

        if ((d4 > self.CommRange)):
            d4 = self.maxDistanceRearVeh
            v4 = -1
        elif d4 < 0: #There is no vehicle behind in L1
            d4 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v4 < 0) and (d4 <= self.CommRange)) : # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v4 = 0

        if ((d5 > self.CommRange)):
            d5 = self.maxDistanceFrontVeh
            v5 = -1
        elif d5 < 0: # no vehicle ahead in L1
            d5 = self.maxDistanceFrontVeh # as this can be considered as vehicle is far away
        if ((v5 < 0) and (d5 <= self.CommRange)) : # there is no vehicle ahead in L1 or there is a communication error: # there is no vehicle ahead in L1
            v5 = 0

        if ((d6 > self.CommRange)):
            d6 = self.maxDistanceRearVeh
            v6 = -1
        elif d6 < 0: #There is no vehicle behind in L1
            d6 = self.maxDistanceRearVeh # so that oue vehicle can go to the overtaking lane
        if ((v6 < 0) and (d6 <= self.CommRange)): # there is no vehicle behind in L1 or there is a communication error: # there is no vehicle behind in L1
            v6 = 0

        # 获取当前车速 va
        va = VehicleParameters[self.AutoCarID][tc.VAR_SPEED]
        # 获取执行前车辆所在车道的纵向位置
        da = VehicleParameters[self.AutoCarID][tc.VAR_LANEPOSITION]
        # 获取执行前车辆前方车辆的纵向位置
        dFront = VehicleParameters[self.AutoCarFrontID][tc.VAR_LANEPOSITION]
        vFront = VehicleParameters[self.AutoCarFrontID][tc.VAR_SPEED]
        # Vehicle acceleration rate 计算速度加速度 vacc。由于时间步长为 1 秒，所以可以用当前速度和上一个时间步长的速度差来计算速将度加速度
        vacc = (va - self.PrevSpeed)/self.delta_t  # as the time step is 1sec long
        # print("d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6:", d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6)
        # 这些参数是用于车辆行驶过程中的决策和控制，例如加速和转向
        return va, da, v1, d1, v2, d2, v3, d3, v4, d4, v5, d5, v6, d6, VehicleParameters[self.AutoCarID][tc.VAR_LANE_INDEX], vacc, dFront, vFront

    # 超车完成标志
    def is_overtake_complete(self, state):
        delta_v = abs(state[0] - state[17])
        overtake_distance = self.ttc_safe * delta_v
        if (state[1] - state[16] - 5) >= overtake_distance:
            self.overpassFlag = 1

        return self.overpassFlag

    # 车流量计算
    def trafficFlowCal(self, state):
        # 目标车辆前方范围
        front_distance_min = 50
        front_distance_max = 150
        front_position_y_min = state + front_distance_min
        front_position_y_max = state + front_distance_max
        # 获取目标车道的车流量
        target_lane0 = 'Lane_0'
        target_lane1 = 'Lane_1'
        target_lane2 = 'Lane_2'
        target_lane0_vehicles = traci.lane.getLastStepVehicleIDs(target_lane0)
        target_lane1_vehicles = traci.lane.getLastStepVehicleIDs(target_lane1)
        target_lane2_vehicles = traci.lane.getLastStepVehicleIDs(target_lane2)
        # 初始化各车道的车流量字典，并设置初始值为0
        lane_traffic = {0: 0, 1: 0, 2: 0}  # key：0车道，1车道，2车道
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        for veh_id in target_lane0_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[0] += 1
        for veh_id in target_lane1_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[1] += 1
        for veh_id in target_lane2_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[2] += 1

        # print("各车道车流量：", lane_traffic)
        return lane_traffic

    # 处理奖励函数，使其各个参数范围相似
    def min_max_normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)
    # 奖励
    def updateReward(self, action, state):
        x = action  # 变道
        L = 100
        t = 0.2
        a_max = 5  # 一般设置在 3 - 5 m/s2

        w4 = 0.5
        w5 = 0.5

        w_f = 1
        w_c = 1
        w_e = 3
        w_s = 2
        w_n = 1

        w_front = 0.5  # 计算前车的TCC权重为0.6

        # calculate lane flow
        Q = self.traFlowNumber
        # print("Q:",Q)
        V = Q / (L * t)  # V表示车道流量，Q表示通过车道的车辆数量，L表示车道的长度，t表示计算流量的时间间隔
        # reward related to lane flow
        r_frontTraffice = 1 - 4 * V    # F为前方车辆的数量

        r_comf = self.min_max_normalize(abs(state[15]), 0, a_max)

        # reward related to efficiency
        # 变道数
        rLaneChange = -3 * self.min_max_normalize(self.numberOfLaneChanges, 0, 50)

        # print("numberOfLaneChanges:",self.numberOfLaneChanges)

        # 速度差异越大，奖励越高，鼓励追赶前方车辆
        if state[0] > state[17]:
            r_v = self.min_max_normalize(state[0] - state[17], 0, 20)
        else:
            r_v = -2 * self.min_max_normalize(state[17] - state[0], 0, 20)

        r_effi_all = w4 * rLaneChange + w5 * r_v

        # reward related to safety
        if x == -1:
            # 左变道计算 TCC
            if state[6] != -1:
                delta_V1 = state[0] - state[6]
                delta_D1 = state[7]
                if delta_V1 < 0:  # 车辆A速度比车辆B快
                    TCC_front = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = delta_D1 / delta_V1
            else:
                TCC_front = 10     # 默认TCC很大，默认为5
            if state[8] != -1:
                delta_V2 = state[0] - state[8]
                delta_D2 = state[9]
                if delta_V2 > 0:
                    TCC_back = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_back = delta_D2 / delta_V2
            else:
                TCC_back = 10     # 默认TCC很大，默认为5
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back  # 前后车的 TCC 是综合计算的

        elif x == 1:
            if state[10] != -1:
                delta_V1 = state[0] - state[10]
                delta_D1 = state[11]
                if delta_V1 < 0:
                    TCC_front = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = delta_D1 / delta_V1
            else:
                TCC_front = 10     # 默认TCC很大，默认为5
            if state[12] != -1:
                delta_V2 = state[0] - state[12]
                delta_D2 = state[13]
                if delta_V2 > 0:
                    TCC_back = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_back = delta_D2 / delta_V2
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back  # 前后车的 TCC 是综合计算的

        else:
            if state[2] != -1:
                delta_V = state[0] - state[2]
                delta_D = state[3]
                if delta_V < 0:  # 车辆A速度比车辆B快
                    TCC_front = 10  # 设置一个较大的值，表示车辆A和车辆B之间的时间间隔很大
                else:
                    TCC_front = delta_D / delta_V
            else:
                TCC_front = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            TCC_surround = TCC_front

        self.finaTCC = TCC_surround

        if TCC_surround <= self.ttc_safe-1:
            # 控制在0-1之间（负奖励）
            r_dis = -1 * self.min_max_normalize(TCC_surround, 0, self.ttc_safe-1)
        else:
            # 控制在0-1之间
            r_dis = 1

        # 如果未完成超车，TTC即使很大，奖励也应该很小
        # 设置调节系数
        adjustment_coefficient = 0.1
        # 判断是否超过了前方目标车辆
        if state[1] > state[16]:
            # 超过前方目标车辆，正常计算TCC值
            r_safe = r_dis
        else:
            r_safe = r_dis * adjustment_coefficient

        isOC = self.is_overtake_complete(state)
        if isOC != 1:
            # 给予固定的负奖励，用于惩罚车辆一直处于未完成超车状态
            r_negative_overtaking = -1
        else:
            r_negative_overtaking = 0  # 如果完成了超车，负奖励为0
            # 创建DataFrame对象
            df = pd.DataFrame({'numberOfLaneChanges': [self.numberOfLaneChanges]})

            # 将数据保存为CSV文件，并去除标题行
            df.to_csv('result/numberOfLaneChanges0820.csv', index=False, mode='a', header=False)

        # total reward
        r_total = w_c * r_comf + w_e * r_effi_all + w_s * r_safe + w_f * r_frontTraffice + w_n * r_negative_overtaking
        # r_total = w_c * r_comf + w_e * r_effi_all + w_s * r_safe + w_n * r_negative_overtaking
        # r_total = w_c * r_comf + w_e * r_effi_all + w_s * r_safe + w_n * r_negative_overtaking  # 暂时不加车流量因素。w_c = 0.2；w_e = 0.3；w_s = 0.5
        # print("r_comf:",w_c * r_comf,";r_effi_all:",w_e * r_effi_all,";r_safe",w_s * r_safe,";r_frontTraffice",w_f * r_frontTraffice,";r_negative_overtaking",w_n * r_negative_overtaking)
        # print("r_comf:", r_comf, ";r_effi_all:", r_effi_all, ";r_safe", r_safe, ";r_frontTraffice",r_frontTraffice)

        return r_total

    def getFinaTCC(self):

        return self.finaTCC



# if __name__ == '__main__':
#     state = (10, 100, 8, 20, 8, 30, 15, 100, 8, 50, -1, -1, 12, 100, 1, 1, 0.5)
#     laneCP = LaneChangePredict()
#     result = laneCP.updateReward(1,state)
#     print(result)


