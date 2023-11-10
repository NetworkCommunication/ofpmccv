import os
import sys
import traci
import traci.constants as tc

sumo_binary = "sumo-gui"  # SUMO的可执行文件路径，如果没有设置环境变量，需要指定完整路径
sumocfg_file = "StraightRoad.sumocfg"  # SUMO配置文件路径
net_file = "StraightRoad.net.xml"  # 路网文件路径
route_file = "StraightRoad.rou.xml"  # 车辆路由文件路径

sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "500", "--scale","1"]
traci.start(sumo_cmd)

# 获取车道数信息
num_lanes = traci.lane.getIDCount()

flag = True
AutoCarID = 'Car'
AutoCarFrontID = 'CarF'
while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()

# vehicle_AllID = list(dict.fromkeys(vehicle_AllID))
# print("道路上的所有车辆 ID：", vehicle_AllID)
# print("*"*10)




traci.close()
