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

## Project structure introduction

Our project is mainly divided into two parts: 1) Predicting the future state of vehicles (corresponding to the predictState file); 2) Using DRL algorithms to control lane-changing overtaking for connected vehicles (corresponding to the Overtaking file).

In the 'predictState' file, "LSTM_behavior_prediction-8.py": predicts the driving behavior of vehicles, "atten_statePrediction_nobehavior.py": combines attention mechanisms and does not consider driving behavior to predict the future state of vehicles, "attention_statePrediction_behavior.py": combines attention mechanisms and considers future driving behaviors to predict the future state of vehicles, "behavior_model.py": defines the neural network structure for driving behavior prediction, "dataFilling.py": fills in missing data, "dataInteractive.py": the interaction model between vehicles, "dataPrepare8.py and processData.py": data preprocessing.

In the 'Overtaking' file, "mainL3.py, mainL4.py, and mainL5.py" correspond to the main running files for three, four, and five lanes respectively, "my_EnvL3.py, my_EnvL4.py, and my_EnvL5.py" correspond to the reinforcement learning environments for three, four, and five lanes respectively. Under the 'data' folder are all the SUMO simulation configuration files, the 'comparison' file is for comparison algorithms, and the 'agents' file is for our proposed algorithms.

## How to Run

In order to make it easier for developers to read, we decompose the code of the state prediction and overtaking model into two independent parts. Run main.py in the predictState file to start, and run main_.py in Overtaking. _Env_.py is the environment code in reinforcement learning. The environment codes under the comparison subfile are all the same, and _Env_.py can be used.

## Statement

In this project, due to the different parameter settings such as the location of the connected vehicle, the location of the target vehicle, and the state of surrounding vehicles, etc., the parameters of reinforcement learning algorithm are set differently, and the reinforcement learning process is different, resulting in different experimental results. In addition, the parameters of the specific network model refer to the parameter settings in the experimental part of the paper. If you want to know more, please refer to our paper "Overtaking Feasibility Prediction for Scenario of Mixed Connected and Connectionless Vehicles".