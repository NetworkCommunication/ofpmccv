import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math
import random
from torch.autograd import Variable
from agents.agent import Agent
from copy import deepcopy
from agents.memory.memory import Memory, ReplayBuffer, NStepReplayBuffer


class QActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(256, 128, 64), activation="relu"):
        super(QActor, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation

        # create layers
        self.layers = nn.ModuleList()
        inputSize = self.state_size + self.action_parameter_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        # self.layers.append(nn.Linear(lastHiddenLayerSize, self.action_parameter_size))
        self.adv_layers = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)
        # self.val_layers = nn.Linear(lastHiddenLayerSize, 1)


    def forward(self, state, action_parameters):
        # implement forward
        negative_slope = 0.01

        x = torch.cat((state, action_parameters.float()), dim=1)
        num_layers = len(self.layers)
        for i in range(0, num_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        # adv1 = self.adv_layers(x)
        # val1 = self.val_layers(x)
        # Q = val1 + adv1 - adv1.mean(dim=1, keepdim=True)
        Q = self.adv_layers(x)
        return Q


class ParamActor(nn.Module):

    def __init__(self, state_size, action_size, action_parameter_size, hidden_layers=(256, 128, 64), activation="relu"):
        super(ParamActor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.action_parameter_size = action_parameter_size
        self.activation = activation
        self.layers = nn.ModuleList()
        inputSize = self.state_size
        lastHiddenLayerSize = inputSize
        if hidden_layers is not None:
            nh = len(hidden_layers)
            self.layers.append(nn.Linear(inputSize, hidden_layers[0]))
            for i in range(1, nh):
                self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            lastHiddenLayerSize = hidden_layers[nh - 1]
        self.action_parameters_output_layer = nn.Linear(lastHiddenLayerSize, self.action_parameter_size)
        self.action_parameters_passthrough_layer = nn.Linear(self.state_size, self.action_parameter_size)

        nn.init.zeros_(self.action_parameters_passthrough_layer.weight)
        nn.init.zeros_(self.action_parameters_passthrough_layer.bias)

        # fix passthrough layer to avoid instability, rest of network can compensate
        self.action_parameters_passthrough_layer.requires_grad = False
        self.action_parameters_passthrough_layer.weight.requires_grad = False
        self.action_parameters_passthrough_layer.bias.requires_grad = False

    def forward(self, state):
        x = state
        negative_slope = 0.01
        num_hidden_layers = len(self.layers)
        for i in range(0, num_hidden_layers):
            if self.activation == "relu":
                x = F.relu(self.layers[i](x))
            elif self.activation == "leaky_relu":
                x = F.leaky_relu(self.layers[i](x), negative_slope)
            else:
                raise ValueError("Unknown activation function " + str(self.activation))
        action_params = self.action_parameters_output_layer(x)
        # print(action_params,"............")
        action_params += self.action_parameters_passthrough_layer(state)
        # print(self.action_parameters_passthrough_layer(state), "、、、、、、、、、、、、、、、")
        return action_params

class PDQNAgent(Agent):
    NAME = "P-DQN Agent"

    def __init__(self,
                 observation_space,
                 action_space,
                 actor_class=QActor,
                 actor_kwargs={},
                 actor_param_class=ParamActor,
                 actor_param_kwargs={},
                 epsilon_initial=1.0,
                 epsilon_final=0.05,
                 epsilon_steps=1000,
                 batch_size=64,
                 gamma=0.99,
                 replay_memory_size=100000,
                 learning_rate_actor=0.0001,
                 learning_rate_actor_param=0.00001,
                 initial_memory_threshold=0,
                 loss_func=F.smooth_l1_loss,  # F.mse_loss
                 clip_grad=10,
                 inverting_gradients=True,
                 device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None):
        super(PDQNAgent, self).__init__(observation_space, action_space)
        self.device = torch.device(device)
        self.num_actions = self.action_space
        self.action_parameter_sizes = np.array(
            [1 for i in range(1, self.num_actions + 1)])
        self.action_parameter_size = int(self.action_parameter_sizes.sum())
        self.action_max = torch.from_numpy(np.ones((self.num_actions,))).float().to(device)
        self.action_min = -self.action_max.detach()
        self.action_range = (self.action_max - self.action_min).detach()
        # print([self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)])
        self.action_parameter_max_numpy = np.concatenate(
            [[3] for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [[1] for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)
        self.action_parameter_max = torch.from_numpy(self.action_parameter_max_numpy).float().to(device)
        self.action_parameter_min = torch.from_numpy(self.action_parameter_min_numpy).float().to(device)
        self.action_parameter_range = torch.from_numpy(self.action_parameter_range_numpy).float().to(device)
        self.epsilon = epsilon_initial
        self.epsilon_initial = epsilon_initial
        self.epsilon_final = epsilon_final
        self.epsilon_steps = epsilon_steps

        self.action_parameter_offsets = self.action_parameter_sizes.cumsum()
        self.action_parameter_offsets = np.insert(self.action_parameter_offsets, 0, 0)

        self.batch_size = batch_size
        self.gamma = gamma
        self.replay_memory_size = replay_memory_size
        self.initial_memory_threshold = initial_memory_threshold
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_actor_param = learning_rate_actor_param
        self.inverting_gradients = inverting_gradients
        self._step = 0
        self._episode = 0
        self.updates = 0
        self.clip_grad = clip_grad

        self.np_random = None
        self.seed = seed
        self._seed(seed)

        # print(self.num_actions + self.action_parameter_size)
        # self.replay_memory = Memory(replay_memory_size, observation_space.shape, (1 + self.action_parameter_size,),
        #                             next_actions=False)
        self.replay_memory = NStepReplayBuffer(capacity=replay_memory_size, seed=self.seed)
        self.actor = actor_class(self.observation_space, self.num_actions, self.action_parameter_size,
                                 **actor_kwargs).to(device)
        self.actor_target = actor_class(self.observation_space, self.num_actions, self.action_parameter_size,
                                        **actor_kwargs).to(device)
        self.actor_target.eval()

        self.actor_param = actor_param_class(self.observation_space, self.num_actions,
                                             self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target = actor_param_class(self.observation_space, self.num_actions,
                                                    self.action_parameter_size, **actor_param_kwargs).to(device)
        self.actor_param_target.eval()

        self.loss_func = loss_func  # l1_smooth_loss performs better but original paper used MSE
        self.actor_optimiser = optim.Adam(self.actor.parameters(),
                                          lr=self.learning_rate_actor)  # , betas=(0.95, 0.999))
        self.actor_param_optimiser = optim.Adam(self.actor_param.parameters(),
                                                lr=self.learning_rate_actor_param)  # , betas=(0.95, 0.999)) #, weight_decay=critic_l2_reg)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.learning_rate_actor) + \
                "Actor Param Alpha: {}\n".format(self.learning_rate_actor_param) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "Initial memory: {}\n".format(self.initial_memory_threshold) + \
                "epsilon_initial: {}\n".format(self.epsilon_initial) + \
                "epsilon_final: {}\n".format(self.epsilon_final) + \
                "epsilon_steps: {}\n".format(self.epsilon_steps) + \
                "Clip Grad: {}\n".format(self.clip_grad) + \
                "Seed: {}\n".format(self.seed) + \
                "epsilon_decay: 1000\n"
        return desc

    def _seed(self, seed=None):
        """
        NOTE: this will not reset the randomly initialised weights; use the seed parameter in the constructor instead.

        :param seed:
        :return:
        """
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        self.np_random = np.random.RandomState(seed=seed)
        if seed is not None:
            torch.manual_seed(seed)
            if self.device == torch.device("cuda"):
                torch.cuda.manual_seed(seed)

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.tensor(initial_bias).float().to(self.device)
            passthrough_layer.requires_grad = False
            passthrough_layer.weight.requires_grad = False
            passthrough_layer.bias.requires_grad = False

        self.actor_param_target.load_state_dict(self.actor_param.state_dict())

    def start_episode(self):
        pass

    def end_episode(self):
        self._episode += 1

        ep = self._episode
        self.epsilon = self.epsilon_final + (self.epsilon_initial - self.epsilon_final) * math.exp(-1. * ep / 1000)
        # if ep>1000:
        #     self.epsilon=self.epsilon_final

    def act(self, state):
        with torch.no_grad():
            state = torch.from_numpy(state).to(self.device)
            all_action_parameters = self.actor_param.forward(state)

            rnd = self.np_random.uniform()
            if rnd < self.epsilon:
                action = self.np_random.choice(self.num_actions)
            else:
                # select maximum action
                Q_a = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_a = Q_a.detach().cpu().data.numpy()
                action = np.argmax(Q_a)

            all_action_parameters = all_action_parameters.cpu().data.numpy()
            offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
            action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def _invert_gradients(self, grad, vals, grad_type, inplace=True):
        if grad_type == "actions":
            max_p = self.action_max
            min_p = self.action_min
            rnge = self.action_range
        elif grad_type == "action_parameters":
            max_p = self.action_parameter_max
            min_p = self.action_parameter_min
            rnge = self.action_parameter_range
        else:
            raise ValueError("Unhandled grad_type: '" + str(grad_type) + "'")

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        rnge = rnge.cpu()
        grad = grad.cpu()
        vals = vals.cpu()

        assert grad.shape == vals.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - vals) / rnge)[index]
            grad[~index] *= ((~index).float() * (vals - min_p) / rnge)[~index]

        return grad

    def _optimize_td_loss(self):
        if self._step < self.batch_size or self._step < self.initial_memory_threshold:
            return
        # Sample a batch from replay memory
        states, actions, rewards, next_states, terminals = self.replay_memory.sample(self.batch_size)

        states = torch.from_numpy(states).to(self.device)
        actions_combined = torch.from_numpy(actions).to(self.device)  # make sure to separate actions and parameters
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.from_numpy(rewards).to(self.device).squeeze()
        next_states = torch.from_numpy(next_states).to(self.device)
        terminals = torch.from_numpy(terminals).to(self.device).squeeze()

        # ---------------------- optimize Q-network ----------------------
        with torch.no_grad():
            pred_next_action_parameters = self.actor_param_target.forward(next_states)
            pred_Q_a = self.actor_target(next_states, pred_next_action_parameters)
            Qprime = torch.max(pred_Q_a, 1, keepdim=True)[0].squeeze()

            # Compute the TD error
            target = rewards + (1 - terminals) * self.gamma * Qprime

        # Compute current Q-values using policy network
        q_values = self.actor(states, action_parameters)
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target
        loss_Q = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_Q.backward()
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.clip_grad)
        self.actor_optimiser.step()

        # ---------------------- optimize actor ----------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True

        Q_val = self.actor(states, action_params)
        Q_loss = torch.mean(torch.sum(Q_val, 1))
        self.actor.zero_grad()
        Q_loss.backward()
        from copy import deepcopy
        delta_a = deepcopy(action_params.grad.data)
        # step 2
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)

        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))
        if self.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), self.clip_grad)

        self.actor_param_optimiser.step()

    def save_models(self, prefix):
        """
        saves the target actor and critic models
        :param prefix: the count of episodes iterated
        :return:
        """
        torch.save(self.actor.state_dict(), prefix + '_actor.pt')
        torch.save(self.actor_param.state_dict(), prefix + '_actor_param.pt')
        print('Models saved successfully')

    def load_models(self, prefix):
        """
        loads the target actor and critic models, and copies them onto actor and critic models
        :param prefix: the count of episodes iterated (used to find the file name)
        :param target: whether to load the target newtwork too (not necessary for evaluation)
        :return:
        """
        # also try load on CPU if no GPU available?
        self.actor.load_state_dict(torch.load(prefix + '_actor.pt', map_location='cpu'))
        self.actor_param.load_state_dict(torch.load(prefix + '_actor_param.pt', map_location='cpu'))
        print('Models loaded successfully')


class PDQNAgent_v1:
    def __init__(self, state_space, action_space, epsilon_initial=1.0, epsilon_final=0.02, epsilon_steps=5000,
                 batch_size=32, gamma=0.90, replay_memory_size=1e6, learning_rate_actor=1e-3,
                 learning_rate_actor_param=1e-4,
                 actor_kwargs={}, actor_param_kwargs={}, device="cuda" if torch.cuda.is_available() else "cpu",
                 seed=None, loss_function=F.smooth_l1_loss, inverting_gradients=True):
        if actor_param_kwargs is None:
            actor_param_kwargs = {}
        self.action_space = action_space
        self.state_space = state_space
        self.device = torch.device(device)
        self.seed = seed
        random.seed(self.seed)
        self.np_random = np.random.RandomState(seed=seed)

        self.num_actions = action_space.spaces[0].n
        self.actions_count = 0
        self.action_parameter_sizes = np.array(
            [self.action_space.spaces[i].shape[0] for i in range(1, 1 + self.num_actions)])
        self.action_parameter_size = self.action_parameter_sizes.sum()
        self.action_parameter_max_numpy = np.concatenate(
            [self.action_space.spaces[i].high for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_min_numpy = np.concatenate(
            [self.action_space.spaces[i].low for i in range(1, self.num_actions + 1)]).ravel()
        self.action_parameter_range_numpy = (self.action_parameter_max_numpy - self.action_parameter_min_numpy)

        self.epsilon = 0
        self.epsilon_start = epsilon_initial
        self.epsilon_end = epsilon_final
        self.epsilon_steps = epsilon_steps

        self.batch_size = batch_size
        self.gamma = gamma
        self.actor_lr = learning_rate_actor
        self.param_net_lr = learning_rate_actor_param
        self.actor = QActor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                            **actor_kwargs).to(self.device)
        self.actor_target = QActor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                                   **actor_kwargs).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.eval()  # 不启用 BatchNormalization 和 Dropout

        self.actor_param = ParamActor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                                      **actor_param_kwargs).to(self.device)
        self.actor_param_target = ParamActor(self.state_space.shape[0], self.num_actions, self.action_parameter_size,
                                             **actor_param_kwargs).to(self.device)
        self.actor_param_target.load_state_dict(self.actor_param.state_dict())
        self.actor_param_target.eval()
        self.inverting_gradients = True

        self.loss_func = loss_function
        self.actor_optimiser = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.param_net_optimiser = optim.Adam(self.actor_param.parameters(), lr=self.param_net_lr)

        self.replay_memory_size = replay_memory_size
        self.replay_memory = ReplayBuffer(capacity=replay_memory_size, seed=self.seed)

    def __str__(self):
        desc = super().__str__() + "\n"
        desc += "Actor Network {}\n".format(self.actor) + \
                "Param Network {}\n".format(self.actor_param) + \
                "Actor Alpha: {}\n".format(self.actor_lr) + \
                "Actor Param Alpha: {}\n".format(self.param_net_lr) + \
                "Gamma: {}\n".format(self.gamma) + \
                "Inverting Gradients: {}\n".format(self.inverting_gradients) + \
                "Replay Memory: {}\n".format(self.replay_memory_size) + \
                "Batch Size: {}\n".format(self.batch_size) + \
                "epsilon_initial: {}\n".format(self.epsilon_start) + \
                "epsilon_final: {}\n".format(self.epsilon_end) + \
                "Seed: {}\n".format(self.seed) + \
                "epsilon_decay: 1000\n"
        return desc

    def set_action_parameter_passthrough_weights(self, initial_weights, initial_bias=None):
        passthrough_layer = self.actor_param.action_parameters_passthrough_layer
        assert initial_weights.shape == passthrough_layer.weight.data.size()
        passthrough_layer.weight.data = torch.tensor(initial_weights).float().to(self.device)
        if initial_bias is not None:
            assert initial_bias.shape == passthrough_layer.bias.data.size()
            passthrough_layer.bias.data = torch.tensor(initial_bias).float().to(self.device)
            passthrough_layer.requires_grad = False
            passthrough_layer.weight.requires_grad = False
            passthrough_layer.bias.requires_grad = False

        self.actor_param_target.load_state_dict(self.actor_param.state_dict())

    def act(self, state, train=True):
        if train:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           math.exp(-1. * self.actions_count / self.epsilon_steps)
            self.actions_count += 1

            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.actor_param.forward(state)

                if random.random() < self.epsilon:
                    action = self.np_random.choice(self.num_actions)
                    all_action_parameters = torch.from_numpy(
                        np.random.uniform(self.action_parameter_min_numpy, self.action_parameter_max_numpy))
                else:
                    Q_value = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                    Q_value = Q_value.detach().cpu().data.numpy()
                    action = np.argmax(Q_value)

                all_action_parameters = all_action_parameters.cpu().data.numpy()
                offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]
        else:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device)
                all_action_parameters = self.actor_param.forward(state)
                Q_value = self.actor.forward(state.unsqueeze(0), all_action_parameters.unsqueeze(0))
                Q_value = Q_value.detach().data.numpy()
                action = Q_value.max(1)[1].item()
                all_action_parameters = all_action_parameters.cpu().data.numpy()
                offset = np.array([self.action_parameter_sizes[i] for i in range(action)], dtype=int).sum()
                action_parameters = all_action_parameters[offset:offset + self.action_parameter_sizes[action]]

        return action, action_parameters, all_action_parameters

    def _optimize_td_loss(self):
        if len(self.replay_memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay_memory.sample(self.batch_size)

        states = torch.tensor(states).to(self.device)
        actions_combined = torch.tensor(actions).to(self.device)
        actions = actions_combined[:, 0].long()
        action_parameters = actions_combined[:, 1:]
        rewards = torch.tensor(rewards).to(self.device).squeeze()
        next_states = torch.tensor(next_states).to(self.device)
        dones = torch.tensor(np.float32(dones)).to(self.device).squeeze()

        # -----------------------optimize Q actor------------------------
        with torch.no_grad():
            next_action_parameters = self.actor_param_target.forward(next_states)
            q_value_next = self.actor_target(next_states, next_action_parameters)
            q_value_max_next = torch.max(q_value_next, 1, keepdim=True)[0].squeeze()

            target = rewards + (1 - dones) * self.gamma * q_value_max_next

        q_values = self.actor(states.float(), action_parameters.float())
        y_predicted = q_values.gather(1, actions.view(-1, 1)).squeeze()
        y_expected = target.float()
        loss_actor = self.loss_func(y_predicted, y_expected)

        self.actor_optimiser.zero_grad()
        loss_actor.backward()
        for param in self.actor.parameters():  # clip防止梯度爆炸
            param.grad.data.clamp_(-1, 1)
        self.actor_optimiser.step()

        # ------------------------optimize param net------------------------------
        with torch.no_grad():
            action_params = self.actor_param(states)
        action_params.requires_grad = True
        q_val = self.actor(states, action_params)
        param_loss = torch.mean(torch.sum(q_val, 1))
        self.actor.zero_grad()
        param_loss.backward()

        delta_a = deepcopy(action_params.grad.data)
        action_params = self.actor_param(Variable(states))
        delta_a[:] = self._invert_gradients(delta_a, action_params, grad_type="action_parameters", inplace=True)
        out = -torch.mul(delta_a, action_params)
        self.actor_param.zero_grad()
        out.backward(torch.ones(out.shape).to(self.device))

        torch.nn.utils.clip_grad_norm_(self.actor_param.parameters(), 10.)

        self.param_net_optimiser.step()

    def _invert_gradients(self, grad, action_params, grad_type, inplace):

        max_p = torch.tensor(self.action_parameter_max_numpy).float().to(self.device)
        min_p = torch.tensor(self.action_parameter_min_numpy).float().to(self.device)
        range = torch.tensor(self.action_parameter_range_numpy).float().to(self.device)

        max_p = max_p.cpu()
        min_p = min_p.cpu()
        range = range.cpu()
        grad = grad.cpu()
        action_params = action_params.cpu()

        assert grad.shape == action_params.shape

        if not inplace:
            grad = grad.clone()
        with torch.no_grad():
            # index = grad < 0  # actually > but Adam minimises, so reversed (could also double negate the grad)
            index = grad > 0
            grad[index] *= (index.float() * (max_p - action_params) / range)[index]
            grad[~index] *= ((~index).float() * (action_params - min_p) / range)[~index]

        return grad

    def save_model(self, path):
        torch.save(self.actor_target.state_dict(), path)
        torch.save(self.actor_param_target.state_dict(), path)

    def load_model(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor_param.load_state_dict(torch.load(path))
