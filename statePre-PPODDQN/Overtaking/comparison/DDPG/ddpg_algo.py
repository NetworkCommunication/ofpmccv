"""
Note: This is a updated version from my previous code,
for the target network, I use moving average to soft replace target parameters instead using assign function.
By doing this, it has 20% speed up on my machine (CPU).

Deep Deterministic Policy Gradient (DDPG), Reinforcement Learning.
DDPG is Actor Critic based algorithm.

Using:
tensorflow 1.14.0
gym 0.15.3
"""
import pandas as pd
import tensorflow as tf
import numpy as np

import time

# from state_normalization import StateNormalization

#####################  hyper parameters  ####################
import traci

from my_Env import LaneChangePredict

MAX_EPISODES = 8000

LR_A = 0.00001
LR_C = 0.00001
GAMMA = 0.999
TAU = 0.01
VAR_MIN = 0.01

MEMORY_CAPACITY = 4000
BATCH_SIZE = 128
OUTPUT_GRAPH = False

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')  # 输入
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        self.sess.run(tf.global_variables_initializer())

        if OUTPUT_GRAPH:
            tf.summary.FileWriter("logs/", self.sess.graph)

    def choose_action(self, s):
        s = np.array(s)
        temp = self.sess.run(self.a, {self.S: s[np.newaxis, :]})
        return temp[0]

    def learn(self):
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 400, activation=tf.nn.relu6, name='l1', trainable=trainable)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound[1], name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 400
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu6(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            net = tf.layers.dense(net, 300, activation=tf.nn.relu6, name='l2', trainable=trainable)
            net = tf.layers.dense(net, 10, activation=tf.nn.relu, name='l3', trainable=trainable)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

sumo_binary = "sumo-gui"
sumocfg_file = "data/StraightRoad.sumocfg"

sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "100", "--scale", "1"]
traci.start(sumo_cmd)

env = LaneChangePredict()
MAX_EP_STEPS = 550
s_dim = env.n_features
a_dim = 2
a_bound = [-1, 1]

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 1
t1 = time.time()
ep_reward_list = []

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0.0
    j = 0
    while j < MAX_EP_STEPS:
        a = ddpg.choose_action(s)

        a = np.clip(np.random.normal(a, var), *a_bound)
        if a[0] >= -1 and a[0] < -1 / 3:
            action = -1
        elif a[0] >= -1 / 3 and a[0] < 1 / 3:
            action = 0
        else:
            action = 1

        action_param = a[1]
        s_, r, is_terminal = env.step(action, action_param)
        ddpg.store_transition(s, a, r, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            ddpg.learn()
        s = s_
        ep_reward += r
        if i == MAX_EPISODES - 1:
            df1 = pd.DataFrame({'speedControl': (np.tanh(action_param) + 1) * 10}, index=[j])
            ttc = env.getFinaTCC()
            df2 = pd.DataFrame({'TCC': ttc}, index=[j])
            df1.to_csv('result/ddpg/speedControl.csv', index=False, mode='a', header=False)
            df2.to_csv('result/ddpg/TTC.csv', index=False, mode='a', header=False)
        j = j + 1
        if j == MAX_EP_STEPS - 1:
            env.writer(0)
        if is_terminal:
            break
    ep_reward_list.append(ep_reward/j)
    print(i, "average_episode_reward：", ep_reward/j)
    df = pd.DataFrame({'Reward': ep_reward_list})

    df.to_csv('result/ddpg/reward.csv', index=False, header=False)
print('Running time: ', time.time() - t1)
