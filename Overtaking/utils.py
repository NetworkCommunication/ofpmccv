import torch
import math
import numpy as np

def discrete_action_user_coupled(action):
    action1 = np.zeros(40)
    for i in range(20):
        action1[i] = action_value_user(action[i])
        action1[i+20] = action_value_user(action[i])
    return action1

def discrete_action_user(action):
    action1 = np.zeros(40)
    for i in range(40):
        action1[i] = action_value_user(action[i])
    return action1

def action_value_user(a):
    if a < 1 and a >= 0.6:
        return 4
    elif a < 0.6 and a >= 0.2:
        return 3
    elif a < 0.2 and a >= -0.2:
        return 2
    elif a < -0.2 and a >= -0.6:
        return 1
    else:
        return 0

def discrete_action(action):
    # print(action)
    action1 = np.zeros(2)
    action1[0] = action_value_direction(action[0])
    action1[1] = action_value_speed(action[1])
    return action1

def action_value_speed(a):
    v_n = (np.tanh(a) + 1) * 10
    return v_n

def action_value_direction(a):
    d = np.tanh(a)
    if d <= -0.5:
        d = -1
    elif d < 0.5 and d > 0.5:
        d = 0
    else:
        d = 1

    return d

def action_value_power(a):
    if a >= -1.0 and a <= 1.0:
        return a + 1
    elif a < -1.0:
        return 0
    elif a > 1.0:
        return 2.0

def get_action(mu, std):
    action = torch.normal(mu, std)
    action = action.data.numpy()
    return action

def log_density(x, mu, std, logstd):
    var = std.pow(2)
    log_density = -(x - mu).pow(2) / (2 * var) \
                  - 0.5 * math.log(2 * math.pi) - logstd
    return log_density.sum(1, keepdim=True)

def flat_grad(grads):
    grad_flatten = []
    for grad in grads:
        grad_flatten.append(grad.view(-1))
    grad_flatten = torch.cat(grad_flatten)
    return grad_flatten

def flat_hessian(hessians):
    hessians_flatten = []
    for hessian in hessians:
        hessians_flatten.append(hessian.contiguous().view(-1))
    hessians_flatten = torch.cat(hessians_flatten).data
    return hessians_flatten

def flat_params(model):
    params = []
    for param in model.parameters():
        params.append(param.data.view(-1))
    params_flatten = torch.cat(params)
    return params_flatten

def update_model(model, new_params):
    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length

def kl_divergence(new_actor, old_actor, states):
    mu, std, logstd = new_actor(torch.Tensor(states))
    mu_old, std_old, logstd_old = old_actor(torch.Tensor(states))
    mu_old = mu_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5
    return kl.sum(1, keepdim=True)


