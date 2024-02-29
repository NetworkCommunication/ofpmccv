import datetime
import os

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

@click.command()
@click.option('--seed', default=0, help='Random seed.', type=int)
@click.option('--episodes', default=8000, help='Number of epsiodes.', type=int)
@click.option('--evaluation-episodes', default=1000, help='Episodes over which to evaluate after training.', type=int)
@click.option('--batch-size', default=16, help='Minibatch size.', type=int)
@click.option('--gamma', default=0.99, help='Discount factor.', type=float)
@click.option('--update-ratio', default=0.1, help='Ratio of updates to samples.', type=float)
@click.option('--inverting-gradients', default=True,
              help='Use inverting gradients scheme instead of squashing function.', type=bool)
@click.option('--initial-memory-threshold', default=1000, help='Number of transitions required to start learning.',
              type=int)
@click.option('--replay-memory-size', default=6000, help='Replay memory size in transitions.', type=int)
@click.option('--epsilon-start', default=0.95, help='Initial epsilon value.', type=float)
@click.option('--epsilon-steps', default=1000, help='Number of episodes over which to linearly anneal epsilon.',
              type=int)
@click.option('--epsilon-final', default=0.02, help='Final epsilon value.', type=float)
@click.option('--learning-rate-actor', default=0.00001, help="Actor network learning rate.", type=float)
@click.option('--learning-rate-actor-param', default=0.00001, help="Critic network learning rate.", type=float)
@click.option('--clip-grad', default=1., help="Gradient clipping.", type=float)
@click.option('--beta', default=0.2, help='Averaging factor for on-policy and off-policy targets.', type=float)
@click.option('--scale-actions', default=True, help="Scale actions.", type=bool)
@click.option('--save-freq', default=0, help='How often to save models (0 = never).', type=int)
@click.option('--save-dir', default="results/soccer", help='Output directory.', type=str)
@click.option('--title', default="PDQN_PDDQN", help="Prefix of output files", type=str)

# Complete the learning of the agent in the environment
def run(seed, episodes, batch_size, gamma, inverting_gradients, initial_memory_threshold, replay_memory_size,
        epsilon_steps, learning_rate_actor, learning_rate_actor_param, title, epsilon_start, epsilon_final, clip_grad,
        beta,
        scale_actions, evaluation_episodes, update_ratio, save_freq, save_dir):

    # Determine whether to generate a model save path
    if save_freq > 0 and save_dir:
        save_dir = os.path.join(save_dir, title + "{}".format(str(seed)))
        os.makedirs(save_dir, exist_ok=True)

    env = LaneChangePredict()
    dir = os.path.join(save_dir, title)
    # Set random seed
    np.random.seed(seed)
    agent_class = PDQNNStepAgent

    agent = agent_class(
        env.n_features, env.n_actions,
        actor_kwargs={
            'activation': "relu", },
        actor_param_kwargs={
            'activation': "relu", },
        batch_size=batch_size,
        learning_rate_actor=learning_rate_actor,
        learning_rate_actor_param=learning_rate_actor_param,
        epsilon_initial=epsilon_start,
        epsilon_steps=epsilon_steps,
        epsilon_final=epsilon_final,
        gamma=gamma,
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
    max_steps = 550
    returns = []
    action_save = []
    param_save = []
    moving_avg_rewards = []
    start_time_train = time.time()

    for i_eps in range(episodes):
        if save_freq > 0 and save_dir and i_eps % save_freq == 0:
            agent.save_models(os.path.join(save_dir, str(i_eps)))
        info = {'status': "NOT_SET"}
        state = env.reset()
        state = np.array(state, dtype=np.float32, copy=False)

        # Returns the prediction result for a given state
        act, act_param, all_action_parameters = agent.act(state)
        # Determine action
        action = env.find_action(act)

        episode_reward = 0.
        action_tmp = []
        param_tmp = []
        transitions = []

        for i_step in range(max_steps):

            next_state, reward, terminal = env.step(action, act_param)

            # Save the speed information of the last episode
            if i_eps == episodes - 1:
                df1 = pd.DataFrame({'speedControl': (np.tanh(act_param.cpu().numpy()) + 1) * 10})
                ttc = env.getFinaTCC()
                df2 = pd.DataFrame({'TCC': ttc},index=[i_step])
                df1.to_csv('result/normal/speedControl.csv', index=False, mode='a', header=False)
                df2.to_csv('result/normal/TTC.csv', index=False, mode='a', header=False)

            next_state = np.array(next_state, dtype=np.float32, copy=False)

            action_tmp.append(action)
            param_tmp.append(act_param)
            next_act, next_act_param, next_all_action_parameters = agent.act(next_state)
            next_action = env.find_action(next_act)
            transitions.append([state, np.concatenate(([act], all_action_parameters.data.cpu())).ravel(), reward,
                                next_state, np.concatenate(([next_act],
                                                            next_all_action_parameters.data.cpu())).ravel(), terminal])


            # Update state and actions
            act, act_param, all_action_parameters = next_act, next_act_param, next_all_action_parameters
            action = next_action

            state = next_state
            episode_reward += reward

            if terminal:
                break
        # Notify the agent that an episode has ended
        agent.end_episode()
        print(i_eps, "average_episode_reward：", episode_reward/i_step)

        n_step_returns = compute_n_step_returns(transitions, gamma)
        for t, nsr in zip(transitions, n_step_returns):
            t.append(nsr)
            agent.replay_memory.append(state=t[0], action_with_param=t[1], reward=t[2], next_state=t[3],
                                       done=t[5], n_step_return=nsr)

        n_updates = int(update_ratio * i_step)
        # Perform TD optimization operations
        for _ in range(n_updates):
            agent._optimize_td_loss()

        if i_eps % 2 == 0:
            agent.actor_target.load_state_dict(agent.actor.state_dict())
            agent.actor_param_target.load_state_dict(agent.actor_param.state_dict())

        returns.append(episode_reward/i_step)
        action_save.append(action_tmp)
        param_save.append(param_tmp)
    plt.plot(returns)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.show()
    df = pd.DataFrame({'Reward': returns})
    df.to_csv('result/normal/reward.csv', index=False, header=False)

    if save_freq > 0 and save_dir:
        agent.save_models(os.path.join(save_dir, str(i_eps)))

    print(agent)

# Compute n-step returns for each transition
def compute_n_step_returns(episode_transitions, gamma):
    n = len(episode_transitions)
    n_step_returns = np.zeros((n,))
    n_step_returns[n - 1] = episode_transitions[n - 1][2]
    for i in range(n - 2, 0, -1):
        reward = episode_transitions[i][2]
        target = n_step_returns[i + 1]
        n_step_returns[i] = reward + gamma * target
    return n_step_returns

if __name__ == '__main__':

    # sumo related startup configuration
    sumo_binary = "sumo-gui"
    sumocfg_file = "data/Lane3/StraightRoad.sumocfg"

    sumo_cmd = [sumo_binary, "-c", sumocfg_file, "--start", "--delay", "100", "--scale", "1"]
    traci.start(sumo_cmd)
    run()



