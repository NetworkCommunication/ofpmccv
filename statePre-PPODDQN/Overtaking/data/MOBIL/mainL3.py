import datetime
import os

# 以允许共享使用不同的MKL库，避免出现不兼容的错误
import traci

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import matplotlib.pyplot as plt
import click
import gym
import pandas as pd

from my_EnvL3 import LaneChangePredict
import numpy as np

from agents.nstep_pdqn import PDQNNStepAgent

# 评估实现的智能体在环境中的表现
def evaluate(env, agent, episodes=1000):
    # 对于每个 episode，记录总收益，步数和达成目标的情况
    returns = []
    timesteps = []
    goals = []

    for i_eps in range(episodes):
        # 重置环境状态，开始新的 episode
        state = env.reset()
        # 设定终止条件，默认为未结束
        terminal = False
        # 初始化 steps 计数器，总奖励
        n_steps = 0
        total_reward = 0.
        # 记录每个 episode 的信息
        info = {'status': "NOT_SET"}
        # 执行策略
        while not terminal:
            # 更新 steps 计数器，获取当前状态
            n_steps += 1
            state = np.array(state, dtype=np.float32, copy=False)
            # 根据当前状态，通过 agent 确定行动值
            act, act_param, all_action_parameters = agent.act(state)
            # 获取相应的行动
            action = env.find_action(act)
            # 执行行动，获取回报、终止状态和新的状态
            state, reward, terminal = env.step(action, act_param)
            # 累计总回报
            total_reward += reward
        print(info['status'])
        # 记录是否达到目标状态
        goal = info['status'] == 'GOAL'
        # 将当前信息添加到相应的列表中
        timesteps.append(n_steps)
        returns.append(total_reward)
        goals.append(goal)
    # 返回总体评估结果，即行动值、步数、目标状态（是否达到目标状态）
    return np.column_stack((returns, timesteps, goals))

# Click 库中的命令行参数选项
@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=1, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=16, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--replay-memory-size', default=5000, help='Replay memory size in transitions.', type=int)  # 500000
@click.option('--epsilon-start', default=0.95, help='Initial epsilon value.', type=float)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.02, help='Final epsilon value.', type=float)
@click.option('--learning-rate-actor', default=0.00001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)  # 1 better than 10.
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)  # 0.5
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
# @click.option('--layers', default=(256,128,64), help='Duplicate action-parameter inputs.')
#               # cls=ClickPythonLiteralOption)
# 保存模型的频率，单位为 episode 数量，0 表示不保存，默认为 0
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/soccer", help='Output directory.', type=str)
@click.option('--title', default="PDQN_PDDQN", help="Prefix of output files", type=str)

# 用于完成智能体在环境中的学习
def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, learning_rate_actor, learning_rate_actor_param, title, epsilon_start, epsilon_final, clip_grad,
        beta,
        scale_actions, evaluation_episodes, update_ratio, save_freq, save_dir):
    # 判断是否生成模型保存路径
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)

    env = LaneChangePredict()
    dir = os.path.join(save_dir, title)
    # env = Monitor(env, directory=os.path.join(dir, str(seed)), video_callable=False, write_upon_reset=False, force=True)
    # 设置随机种子，获取环境实例
    np.random.seed(seed)
    # 初始化 agent
    agent_class = PDQNNStepAgent

    agent = agent_class(
        env.n_features, env.n_actions,
        actor_kwargs={
            'activation': "relu", },
        actor_param_kwargs={
            'activation': "relu", },
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,  # 0.0001
        learning_rate_actor_param=learning_rate_actor_param,  # 0.001
        epsilon_initial=epsilon_start,
        epsilon_steps=epsilon_steps,
        epsilon_final=epsilon_final,
        gamma=gamma,  # 0.99
        clip_grad=clip_grad,
        beta=beta,
        initial_memory_threshold=initial_memory_threshold,
        replay_memory_size=replay_memory_size,
        inverting_gradients=inverting_gradients,
        seed=seed)
    print(agent)
    network_trainable_parameters = sum(p.numel() for p in agent.actor.parameters() if p.requires_grad)
    network_trainable_parameters += sum(p.numel() for p in agent.actor_param.parameters() if p.requires_grad)
    print("Total Trainable Network Parameters: %d" % network_trainable_parameters)
    # 训练智能体
    max_steps = 15000   # 原本为15000
    returns = []
    action_save = []
    param_save = []
    moving_avg_rewards = []
    start_time_train = time.time()

    for i_eps in range(episodes):
        # 如果需要存储中间模型，当前回合计数符合要求时执行，并将模型保存到指定路径下
        if save_freq > 0 and save_dir and i_eps % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i_eps)))
        # 重置 episode 的信息，状态和行动
        info = {'status': "NOT_SET"}
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)
        # print("state shape:", state.shape)   # (17,)

        # act()函数返回的是参数化DQN模型对给定状态的预测结果（包括横向离散行动和纵向连续行动），即act()函数的返回值是实际行动之前的预测，也就是选择一个最高价值的行动。
        act, act_param, all_action_parameters = agent.act(state)
        # 使用 find_action 方法将行动值转化为对应的行动，用于实际执行操作
        action = env.find_action(act)
        # print("action:", action)
        # 初始化每个回合的总数回报值为0，并创建用于存储事件转换和当前环境中采取的行动、行动参数值以及回报值对应的列表变量
        episode_reward = 0.
        action_tmp = []
        param_tmp = []
        transitions = []
        # 开始一个 episode 的循环
        for i_step in range(max_steps):
            # print(act,"........................")
            # 根据当前状态、行动值和 agent 选择规则，选择下一行动值以及计算其奖励情况；调用环境的step()函数，该函数会返回一个新状态和这个行动的回报
            next_state, speed, terminal = env.step(action, act_param)
            # 保存最后一轮的速度信息
            if i_eps == episodes - 1:
                df1 = pd.DataFrame({'speedControl': speed},index=[i_step])
                ttc = env.getFinaTCC()
                df2 = pd.DataFrame({'TCC': ttc},index=[i_step])
                df1.to_csv('result/speedControl0820.csv', index=False, mode='a', header=False)
                df2.to_csv('result/TTC0820.csv', index=False, mode='a', header=False)
            # print(reward, "........................")
            next_state = np.array(next_state, dtype=np.float32, copy=False)
            # 记录行动和相关参数，并将状态、行动和奖励、下一个状态添加到当前 episode 的经验记忆中
            action_tmp.append(action)
            param_tmp.append(act_param)
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = env.find_action(next_act)

            # 更新状态和行动，累加总回报，并检查 episode 是否结束
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action
            # print("action:", action)
            state = next_state
            # print("transitions", transitions[0][2])
            if terminal:
                # print("提前停止！")
                break
            # 通知 agent 结束了一个 episode
        agent.end_episode()
        # 平均每步的回报
        print(i_eps, "average_episode_reward：", episode_reward/i_step)
        # print(i_eps, "episode_reward：", episode_reward)

        # calculate n-step returns
        # n_step_returns = compute_n_step_returns(transitions, gamma)
        # 将上面存储的信息添加到记忆中
        # for t, nsr in zip(transitions, n_step_returns):
        #     t.append(nsr)
        #     agent.replay_memory.append(state=t[0], action_with_param=t[1], reward=t[2], next_state=t[3],
        #                                done=t[5], n_step_return=nsr)

        n_updates = int(update_ratio * i_step)
        # 执行 TD 误差函数的优化操作，将更新应用到神经网络参数中
        for _ in range(n_updates):
            agent._optimize_td_loss()

        # 如果当前回合数为偶数，执行带有滑动平均的目标网络更新，即使用当前的 actor 和 actor_param 网络更新每个目标网络。
        # 在此期间，记录动作和参数值，并将该回合的回报值添加到返回变量中
        if i_eps % 2 == 0:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.actor_param_target.load_state_dict(agent.actor_param.state_dict())

        # 将当前回合的总回报添加到存储总回报的列表中
        returns.append(episode_reward/i_step)
        action_save.append(action_tmp)
        param_save.append(param_tmp)
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    # 创建DataFrame对象
    df = pd.DataFrame({'Reward': returns})

    # 将数据保存为CSV文件
    df.to_csv('result/reward0820.csv', index=False, header=False)
    # np.save(file="data/11/action_save.npy", arr=action_save)
    # np.save(file="data/11/param_save.npy", arr=param_save)

    # end_time_train = time.time()
    if save_freq > 0 and save_dir:
        # 将模型保存到指定的路径中
        agent.save_models(os.path.join(save_dir, str(i_eps)))

    print(agent)

# 计算每个转换的 n 步回报。第一个表示本轮训练中转换事件的转换集合、第二个表示回报衰减因子
def compute_n_step_returns(episode_transitions, gamma):
    # 计算该集合的长度并创建一个形状为 n 的零数组，用于存储计算出的每个转换的 n 步回报
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    # 将最后一个转换的 n 步回报设置为该转换的回报值，因为对于该转换而言，最后一步后没有更多的行动
    n_step_returns[n - 1] = episode_transitions[n - 1][2]  # Q-value is just the final reward
    # 通过倒序循环来计算集合中的所有其他转换的 n 步回报
    for i in range(n - 2, 0, -1):
        # 为变量 reward 分配本次迭代中处理的转换的回报值，并为变量 target 分配下一步转换的 Q 值
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        # 通过计算reward + gamma * target，将该步转换的n步回报更新为该数组中的当前位置
        # 因为是倒序迭代，索引为i + 1的n步返回值被视为当前的值，因此每次迭代都会向前平移整个集合，
        # 并将更新的值赋值给n_step_returns数组中索引为i的位置
        n_step_returns[i] = reward + gamma * target
    # 返回计算的 n 步回报数组
    return n_step_returns

if __name__ == '__main__':
    sumo_binary = "sumo-gui"  # SUMO的可执行文件路径，如果没有设置环境变量，需要指定完整路径
    sumocfg_file = "StraightRoad.sumocfg"  # SUMO配置文件路径

    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "100", "--scale", "1"]
    traci.start(sumo_cmd)
    run()



