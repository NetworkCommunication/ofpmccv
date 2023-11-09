# statePre-PPODDQN

## Project Introduction

Our system has two main components: predicting vehicle states and evaluating the feasibility of lane-changing overtaking. This involves three key neural network models. First, a driving behavior prediction model anticipates future vehicle actions. Next, these predictions, along with vehicle state and other data, inform a state prediction model to precisely forecast future vehicle states. These predicted states are then used to initialize the PPODDQN model, which simulates lane-changing overtaking maneuvers. Ultimately, we compute the Feasibility Assessment Index (FAI), which considers factors like speed guidance feasibility, risk assessment, required lane changes, and prediction accuracy. High FAI values prompt the vehicle to initiate overtaking, with the system providing corresponding speed guidance.

## Environmental Dependence

The code requires python3 (>=3.8) with the development headers. The code also need system packages as bellow:

numpy == 1.24.3

matplotlib == 3.7.1

pandas == 1.5.3

pytorch == 2.0.0

gym == 0.22.0

sumolib == 1.17.0

traci == 1.16.0

If users encounter environmental problems and reference package version problems that prevent the program from running, please refer to the above installation package and corresponding version.

## How to Run

In order to make it easier for developers to read, we decompose the code of the state prediction and overtaking model into two independent parts. Run main.py in the predictState file to start, and run main_.py in Overtaking. _Env_.py is the environment code in reinforcement learning. The environment codes under the comparison subfile are all the same, and _Env_.py can be used.

## Statement

In this project, due to the different parameter settings such as the location of the connected vehicle, the location of the target vehicle, and the state of surrounding vehicles, etc., the parameters of reinforcement learning algorithm are set differently, and the reinforcement learning process is different, resulting in different experimental results. In addition, the parameters of the specific network model refer to the parameter settings in the experimental part of the paper. If you want to know more, please refer to our paper "Overtaking Feasibility Prediction for Scenario of Mixed Connected and Connectionless Vehicles".