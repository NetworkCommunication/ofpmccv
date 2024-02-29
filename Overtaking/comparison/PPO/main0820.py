import time
import torch
import argparse
import numpy as np
import torch.optim as optim
import traci

from model import Actor, Critic
from my_Env import LaneChangePredict
from utils import get_action
from utils import discrete_action
from utils import discrete_action_user
from collections import deque
from hparams import HyperParams as hp
import matplotlib.pyplot as plt
import Env_UAV_network
import os
import pandas as pd
import seaborn as sns
import time
from tqdm import tqdm
from tqdm._tqdm import trange
import warnings
warnings.filterwarnings("ignore", category=Warning)

sns.set(style='whitegrid', color_codes=True)
parser = argparse.ArgumentParser()
parser.add_argument('--algorithm', type=str, default='PPO',
                    help='select one of algorithms among Vanilla_PG, NPG, TPRO, PPO')
parser.add_argument('--env', type=str, default="Humanoid-v2",
                    help='name of Mujoco environement')
parser.add_argument('--render', default=False)
args = parser.parse_args()

if args.algorithm == "PG":
    from vanila_pg import train_model
elif args.algorithm == "NPG":
    from npg import train_model
elif args.algorithm == "TRPO":
    from trpo import train_model
elif args.algorithm == "PPO":
    from ppo import train_model

if __name__ == "__main__":

    sumo_binary = "sumo-gui"
    sumocfg_file = "data/StraightRoad.sumocfg"

    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "10", "--scale", "1"]
    traci.start(sumo_cmd)

    N_UAV = 1

    num_input = 18
    num_output = 2

    actors = []
    critics = []
    actor_optims = []
    critic_optims = []

    actor = Actor(num_input, num_output)
    critic = Critic(num_input)

    actor_optim = optim.Adam(actor.parameters(), lr=hp.actor_lr1)
    critic_optim = optim.Adam(critic.parameters(), lr=hp.critic_lr1, weight_decay=hp.l2_rate)

    actors.append(actor)
    critics.append(critic)
    actor_optims.append(actor_optim)
    critic_optims.append(critic_optim)

    actor1 = Actor(num_input, num_output)
    critic1 = Critic(num_input)
    actor_optim1 = optim.Adam(actor1.parameters(), lr=hp.actor_lr)
    critic_optim1 = optim.Adam(critic1.parameters(), lr=hp.critic_lr, weight_decay=hp.l2_rate)

    actors.append(actor1)
    critics.append(critic1)
    actor_optims.append(actor_optim1)
    critic_optims.append(critic_optim1)

    episodes = 0
    xar = []
    yar = []
    best_score = 0
    max_episodes = 8000
    max_steps = 550
    env = LaneChangePredict()
    for iter in range(max_episodes):
        memorys = []
        for i in range(N_UAV):
            actors[i].eval(), critics[i].eval()
            memory = deque()
            memorys.append(memory)

        scores = []

        score = 0

        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        for i_step in range(max_steps):
            action_all = np.zeros([N_UAV, 3])
            action_all_idx = 0
            s_all = []
            a_all = []
            for i in range(N_UAV):

                s_all.append(state)
                mu, std, _ = actors[i](torch.Tensor(state).unsqueeze(0))
                action = get_action(mu, std)[0]
                action_discrete = discrete_action(action)

                a_all.append(action)
                action_all[action_all_idx, 0] = action_discrete[0]
                action_all[action_all_idx, 1] = action_discrete[1]
                action1 = action_discrete[0]
                action2 = action_discrete[1]

            s_all.append(state)

            next_state, reward, terminal = env.step(action1, action2)

            if iter == max_episodes - 1:
                df1 = pd.DataFrame({'speedControl': action2},index=[i_step])
                ttc = env.getFinaTCC()
                df2 = pd.DataFrame({'TCC': ttc}, index=[i_step])
                df1.to_csv('result/ppo/speedControl.csv', index=False, mode='a', header=False)
                df2.to_csv('result/ppo/TTC.csv', index=False, mode='a', header=False)

            done = 1
            if i_step == max_steps - 1:
                env.writer(0)

            if terminal:
                done = 0
            for i in range(N_UAV):
                s = s_all[i]
                a = a_all[i]
                memorys[i].append([s, a, reward, done])

            state = next_state

            score += reward
            if done == 0:
                break



        scores.append(score/i_step)

        print(iter, "average_episode_reward：", scores)

        xar.append(int(episodes))
        yar.append(scores)
        start_train = time.time()
        for i in range(N_UAV):
            actors[i].train(), critics[i].train()
            train_model(actors[i], critics[i], memorys[i], actor_optims[i], critic_optims[i])

        df = pd.DataFrame({'Reward': scores})
        df.to_csv('result/ppo/reward.csv', index=False, mode='a', header=False)



