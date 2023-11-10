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
        __file__), '..', '..', '..', '..', "tools"))
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(
        os.path.dirname(__file__), "..", "..", "..")), "tools"))
    from sumolib import checkBinary
except ImportError:
    sys.exit(
        "please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")

gui = False
if gui:
    sumoBinary = checkBinary('sumo-gui')
else:
    sumoBinary = checkBinary('sumo')

config_path = "data/Lane5/StraightRoad.sumocfg"

class LaneChangePredict(gym.Env):
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
        self.maxLaneNumber = 4

        self.CommRange = 150

        self.delta_t = 0.2
        self.AutoCarID = 'Car'
        self.PrevSpeed = 0
        self.PrevVehDistance = 0
        self.VehicleIds = 0
        self.traFlowNumber = 0
        self.finaTCC = 0

        self.overpassFlag = 0
        self.AutoCarFrontID = 'CarF'
        self.ttc_safe = 3


        self.action_space_vehicle =[-1, 0, 1]
        self.n_actions = len(self.action_space_vehicle)
        self.n_actions = int(self.n_actions)

        self.param_velocity = [0, 20]
        self.n_features = 18

        self.actions = np.array([[0, -1], [1, 0], [2, 1]])



    def reset(self):
        self.TotalReward = 0
        self.numberOfLaneChanges = 0
        self.numberOfOvertakes = 0
        self.currentTrackingVehId = 'None'
        self.overpassFlag = 0

        traci.close()

        sumo_binary = "sumo"
        sumocfg_file = "data/Lane5/StraightRoad.sumocfg"

        sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--delay", "100", "--scale", "1"]
        traci.start(sumo_cmd)

        print('Resetting the layout')
        traci.simulationStep()

        self.VehicleIds = traci.vehicle.getIDList()

        for veh_id in self.VehicleIds:
            traci.vehicle.subscribe(veh_id, [tc.VAR_LANE_INDEX, tc.VAR_LANEPOSITION, tc.VAR_SPEED, tc.VAR_ACCELERATION])

        self.state = self._findstate()

        return np.array(self.state)

    def find_action(self, index):
        return self.actions[index][1]

    def step(self, action, action_param):
        x = action
        v_n = (np.tanh(action_param.cpu().numpy()) + 1) * 10
        desired_speed = float(v_n.item())
        Vehicle_Params = traci.vehicle.getAllSubscriptionResults()
        self.PrevSpeed = Vehicle_Params[self.AutoCarID][tc.VAR_SPEED]
        self.PrevVehDistance = Vehicle_Params[self.AutoCarID][tc.VAR_LANEPOSITION]

        traci.vehicle.setSpeed(self.AutoCarID, desired_speed)

        if x == 1:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            if laneindex != 0:
                traci.vehicle.changeLane(self.AutoCarID, laneindex - 1, 100)
                self.numberOfLaneChanges += 1
                self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex - 1]
        elif x == -1:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            if laneindex != self.maxLaneNumber:
                traci.vehicle.changeLane(self.AutoCarID, laneindex + 1, 100)
                self.numberOfLaneChanges += 1
                self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex + 1]
        else:
            laneindex = traci.vehicle.getSubscriptionResults(self.AutoCarID)[tc.VAR_LANE_INDEX]
            self.traFlowNumber = self.trafficFlowCal(self.state[1])[laneindex]

        traci.simulationStep()

        self.state = self._findstate()

        self.end = self.is_overtake_complete(self.state)

        reward = self.updateReward(action, self.state)

        return self.state, reward, self.end


    def close(self):
        traci.close()

    def _findRearVehDistance(self, vehicleparameters):
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
        for VehID in self.VehicleIds:
            parameters[i][0] = VehID
            parameters[i][1] = vehicleparameters[VehID][tc.VAR_LANEPOSITION]
            parameters[i][2] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]
            parameters[i][3] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]
            parameters[i][4] = vehicleparameters[VehID][tc.VAR_LANE_INDEX]
            i = i + 1

        parameters = sorted(parameters, key=lambda x: x[1])  # Sorted in ascending order based on x distance

        index = [x for x in parameters if self.AutoCarID in x][0]
        RowIDAuto = parameters.index(index)

        if RowIDAuto == len(self.VehicleIds) - 1:
            d1 = -1
            v1 = -1
            d3 = -1
            v3 = -1
            d5 = -1
            v5 = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = 'None'
        else:
            if parameters[RowIDAuto][2] == 0:
                d5 = -1
                v5 = -1
                d6 = -1
                v6 = -1
            elif parameters[RowIDAuto][2] == (self.maxLaneNumber - 1):
                d3 = -1
                v3 = -1
                d4 = -1
                v4 = -1
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d1 = parameters[index][1] - parameters[RowIDAuto][1]
                    v1 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            if index == len(self.VehicleIds):
                d1 = -1
                v1 = -1
                self.CurrFrontVehID = 'None'
                self.CurrFrontVehDistance = 150
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] + 1):
                    d3 = parameters[index][1] - parameters[RowIDAuto][1]
                    v3 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            if index == len(self.VehicleIds):
                d3 = -1
                v3 = -1
            index = RowIDAuto + 1
            while index != len(self.VehicleIds):
                if parameters[index][2] == (parameters[RowIDAuto][2] - 1):
                    d5 = parameters[index][1] - parameters[RowIDAuto][1]
                    v5 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index += 1
            if index == len(self.VehicleIds):
                d5 = -1
                v5 = -1
            index = RowIDAuto - 1
            while index >= 0:
                if parameters[index][2] == parameters[RowIDAuto][2]:
                    d2 = parameters[RowIDAuto][1] - parameters[index][1]
                    v2 = vehicleparameters[parameters[index][0]][tc.VAR_SPEED]
                    break
                index -= 1
            if index < 0:
                d2 = -1
                v2 = -1
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
            if (self.currentTrackingVehId != 'None' and (
                    vehicleparameters[self.currentTrackingVehId][tc.VAR_LANEPOSITION] <
                    vehicleparameters[self.AutoCarID][tc.VAR_LANEPOSITION])):
                self.numberOfOvertakes += 1
            self.currentTrackingVehId = parameters[RowIDAuto + 1][0]
        if RowIDAuto == 0:
            RearDist = -1
        else:
            RearDist = (parameters[RowIDAuto][1] - parameters[RowIDAuto - 1][
                1])
        if RowIDAuto == len(self.VehicleIds) - 1:
            FrontDist = -1
            self.CurrFrontVehID = 'None'
            self.CurrFrontVehDistance = 150
        else:
            FrontDist = (parameters[RowIDAuto + 1][1] - parameters[RowIDAuto][
                1])
            self.CurrFrontVehID = parameters[RowIDAuto + 1][0]
            self.CurrFrontVehDistance = FrontDist
        return d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6

    def _findstate(self):
        VehicleParameters = traci.vehicle.getAllSubscriptionResults()
        d1, v1, d2, v2, d3, v3, d4, v4, d5, v5, d6, v6 = self._findRearVehDistance(VehicleParameters)
        if ((d1 > self.CommRange)):
            d1 = self.maxDistanceFrontVeh
            v1 = -1
        elif d1 < 0:
            d1 = self.maxDistanceFrontVeh
        if ((v1 < 0) and (d1 <= self.CommRange)):
            v1 = 0

        if ((d2 > self.CommRange)):
            d2 = self.maxDistanceRearVeh
            v2 = -1
        elif d2 < 0:
            d2 = 0
        if ((v2 < 0) and (d2 <= self.CommRange)):
            v2 = 0
        if ((d3 > self.CommRange)):
            d3 = self.maxDistanceFrontVeh
            v3 = -1
        elif d3 < 0:
            d3 = self.maxDistanceFrontVeh
        if ((v3 < 0) and (d3 <= self.CommRange)) :
            v3 = 0

        if ((d4 > self.CommRange)):
            d4 = self.maxDistanceRearVeh
            v4 = -1
        elif d4 < 0:
            d4 = self.maxDistanceRearVeh
        if ((v4 < 0) and (d4 <= self.CommRange)) :
            v4 = 0

        if ((d5 > self.CommRange)):
            d5 = self.maxDistanceFrontVeh
            v5 = -1
        elif d5 < 0:
            d5 = self.maxDistanceFrontVeh
        if ((v5 < 0) and (d5 <= self.CommRange)) :
            v5 = 0

        if ((d6 > self.CommRange)):
            d6 = self.maxDistanceRearVeh
            v6 = -1
        elif d6 < 0:
            d6 = self.maxDistanceRearVeh
        if ((v6 < 0) and (d6 <= self.CommRange)):
            v6 = 0

        va = VehicleParameters[self.AutoCarID][tc.VAR_SPEED]
        da = VehicleParameters[self.AutoCarID][tc.VAR_LANEPOSITION]
        dFront = VehicleParameters[self.AutoCarFrontID][tc.VAR_LANEPOSITION]
        vFront = VehicleParameters[self.AutoCarFrontID][tc.VAR_SPEED]
        vacc = (va - self.PrevSpeed)/self.delta_t
        return va, da, v1, d1, v2, d2, v3, d3, v4, d4, v5, d5, v6, d6, VehicleParameters[self.AutoCarID][tc.VAR_LANE_INDEX], vacc, dFront, vFront

    def is_overtake_complete(self, state):
        delta_v = abs(state[0] - state[17])
        overtake_distance = self.ttc_safe * delta_v
        if (state[1] - state[16] - 5) >= overtake_distance:
            self.overpassFlag = 1

        return self.overpassFlag

    def trafficFlowCal(self, state):
        front_distance_min = 50
        front_distance_max = 150
        front_position_y_min = state + front_distance_min
        front_position_y_max = state + front_distance_max
        target_lane0 = 'Lane_0'
        target_lane1 = 'Lane_1'
        target_lane2 = 'Lane_2'
        target_lane3 = 'Lane_3'
        target_lane4 = 'Lane_4'
        target_lane0_vehicles = traci.lane.getLastStepVehicleIDs(target_lane0)
        target_lane1_vehicles = traci.lane.getLastStepVehicleIDs(target_lane1)
        target_lane2_vehicles = traci.lane.getLastStepVehicleIDs(target_lane2)
        target_lane3_vehicles = traci.lane.getLastStepVehicleIDs(target_lane3)
        target_lane4_vehicles = traci.lane.getLastStepVehicleIDs(target_lane4)
        lane_traffic = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
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
        for veh_id in target_lane3_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[3] += 1
        for veh_id in target_lane4_vehicles:
            y = VehicleParameters[veh_id][tc.VAR_LANEPOSITION]
            if y >= front_position_y_min and y <= front_position_y_max:
                lane_traffic[4] += 1

        return lane_traffic

    def min_max_normalize(self, value, min_value, max_value):
        return (value - min_value) / (max_value - min_value)

    def updateReward(self, action, state):
        x = action
        L = 100
        t = 0.2
        a_max = 5

        w4 = 0.5
        w5 = 0.5

        w_f = 1
        w_c = 1
        w_e = 3
        w_s = 2
        w_n = 1

        w_front = 0.5

        # calculate lane flow
        Q = self.traFlowNumber
        V = Q / (L * t)
        # reward related to lane flow
        r_frontTraffice = 1 - 4 * V

        r_comf = self.min_max_normalize(abs(state[15]), 0, a_max)

        # reward related to efficiency
        rLaneChange = -3 * self.min_max_normalize(self.numberOfLaneChanges, 0, 50)

        if state[0] > state[17]:
            r_v = self.min_max_normalize(state[0] - state[17], 0, 20)
        else:
            r_v = -2 * self.min_max_normalize(state[17] - state[0], 0, 20)

        r_effi_all = w4 * rLaneChange + w5 * r_v

        # reward related to safety
        if x == -1:
            if state[6] != -1:
                delta_V1 = state[0] - state[6]
                delta_D1 = state[7]
                if delta_V1 < 0:
                    TCC_front = 10
                else:
                    TCC_front = delta_D1 / delta_V1
            else:
                TCC_front = 10
            if state[8] != -1:
                delta_V2 = state[0] - state[8]
                delta_D2 = state[9]
                if delta_V2 > 0:
                    TCC_back = 10
                else:
                    TCC_back = delta_D2 / delta_V2
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back

        elif x == 1:
            if state[10] != -1:
                delta_V1 = state[0] - state[10]
                delta_D1 = state[11]
                if delta_V1 < 0:
                    TCC_front = 10
                else:
                    TCC_front = delta_D1 / delta_V1
            else:
                TCC_front = 10
            if state[12] != -1:
                delta_V2 = state[0] - state[12]
                delta_D2 = state[13]
                if delta_V2 > 0:
                    TCC_back = 10
                else:
                    TCC_back = delta_D2 / delta_V2
            else:
                TCC_back = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            if abs(TCC_back) > 10:
                TCC_back = 10
            TCC_surround = w_front * TCC_front + (1 - w_front) * TCC_back

        else:
            if state[2] != -1:
                delta_V = state[0] - state[2]
                delta_D = state[3]
                if delta_V < 0:
                    TCC_front = 10
                else:
                    TCC_front = delta_D / delta_V
            else:
                TCC_front = 10
            if abs(TCC_front) > 10:
                TCC_front = 10
            TCC_surround = TCC_front

        self.finaTCC = TCC_surround

        if TCC_surround <= self.ttc_safe-1:
            r_dis = -1 * self.min_max_normalize(TCC_surround, 0, self.ttc_safe-1)
        else:
            r_dis = 1

        adjustment_coefficient = 0.1
        if state[1] > state[16]:
            r_safe = r_dis
        else:
            r_safe = r_dis * adjustment_coefficient

        isOC = self.is_overtake_complete(state)
        if isOC != 1:
            r_negative_overtaking = -1
        else:
            r_negative_overtaking = 0
            df = pd.DataFrame({'numberOfLaneChanges': [self.numberOfLaneChanges]})

            df.to_csv('result/other/numberOfLaneChanges0820.csv', index=False, mode='a', header=False)

        # total reward
        r_total = w_c * r_comf + w_e * r_effi_all + w_s * r_safe + w_f * r_frontTraffice + w_n * r_negative_overtaking
        # r_total = w_c * r_comf + w_e * r_effi_all + w_s * r_safe + w_n * r_negative_overtaking

        return r_total

    def getFinaTCC(self):

        return self.finaTCC



