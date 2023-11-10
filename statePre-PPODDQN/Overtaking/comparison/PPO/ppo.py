import numpy as np
from utils import *
from hparams import HyperParams as hp

def get_gae(rewards, masks, values):
    rewards = torch.Tensor(rewards)
    masks = torch.Tensor(masks)
    returns = torch.zeros_like(rewards)
    advants = torch.zeros_like(rewards)

    running_returns = 0
    previous_value = 0
    running_advants = 0

    for t in reversed(range(0, len(rewards))):
        running_returns = rewards[t] + hp.gamma * running_returns * masks[t]
        running_tderror = rewards[t] + hp.gamma * previous_value * masks[t] - \
                    values.data[t]
        running_advants = running_tderror + hp.gamma * hp.lamda * \
                          running_advants * masks[t]

        returns[t] = running_returns
        previous_value = values.data[t]
        advants[t] = running_advants

    advants = (advants - advants.mean()) / advants.std()
    return returns, advants

def surrogate_loss(actor, advants, states, old_policy, actions, index):
    mu, std, logstd = actor(torch.Tensor(states))
    new_policy = log_density(actions, mu, std, logstd)
    old_policy = old_policy[index]

    ratio = torch.exp(new_policy - old_policy)
    surrogate = ratio * advants
    return surrogate, ratio

def train_model(actor, critic, memory, actor_optim, critic_optim):
    memory = np.array(memory)
    states = [item[0] for item in memory]
    actions = [item[1] for item in memory]
    rewards = [item[2] for item in memory]
    masks = [item[3] for item in memory]

    values = critic(torch.Tensor(states))

    returns, advants = get_gae(rewards, masks, values)
    mu, std, logstd = actor(torch.Tensor(states))
    old_policy = log_density(torch.Tensor(actions), mu, std, logstd)

    criterion = torch.nn.MSELoss()
    n = len(states)
    arr = np.arange(n)

    for epoch in range(10):
        np.random.shuffle(arr)

        for i in range(n // hp.batch_size):
            batch_index = arr[hp.batch_size * i: hp.batch_size * (i + 1)]
            batch_index = torch.LongTensor(batch_index)
            inputs = torch.Tensor(states)[batch_index]
            returns_samples = returns.unsqueeze(1)[batch_index]
            advants_samples = advants.unsqueeze(1)[batch_index]
            actions_samples = torch.Tensor(actions)[batch_index]

            loss, ratio = surrogate_loss(actor, advants_samples, inputs,
                                         old_policy.detach(), actions_samples,
                                         batch_index)

            values = critic(inputs)
            critic_loss = criterion(values, returns_samples)
            critic_optim.zero_grad()
            critic_loss.backward()
            critic_optim.step()
            clipped_ratio = torch.clamp(ratio,
                                        1.0 - hp.clip_param,
                                        1.0 + hp.clip_param)
            clipped_loss = clipped_ratio * advants_samples

            actor_loss = -torch.min(loss, clipped_loss).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()







