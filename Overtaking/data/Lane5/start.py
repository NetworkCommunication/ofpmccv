import os
import sys
import traci
import traci.constants as tc

sumo_binary = "sumo-gui"
sumocfg_file = "StraightRoad.sumocfg"
net_file = "StraightRoad.net.xml"
route_file = "StraightRoad.rou.xml"

sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "500", "--scale", "1"]
traci.start(sumo_cmd)

num_lanes = traci.lane.getIDCount()

flag = True
AutoCarID = 'Car'
AutoCarFrontID = 'CarF'
while traci.simulation.getMinExpectedNumber() > 0:

    traci.simulationStep()




traci.close()
