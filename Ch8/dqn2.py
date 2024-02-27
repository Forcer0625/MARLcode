import math
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DeepQNetwork(nn.Module):

    def __init__(self, n_observations, layer1_dims, layer2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.layer1 = nn.Linear(n_observations, layer1_dims)
        self.layer2 = nn.Linear(layer1_dims, layer2_dims)
        self.layer3 = nn.Linear(layer2_dims, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class DQN():
    def __init__(self, env, config):
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = config['batch_size']
        self.gamma = config['gamma']
        self.eps_start = config['eps_start']
        self.eps_end = config['eps_end']
        self.eps_dec = config['eps_dec']
        self.tau = config['tau']
        self.lr = config['lr']
        self.env = env
        self.layer_dims = config['layer_dims']
        self.n_actions = env.action_space.n
        state, info = env.reset()
        self.n_observations = len(state)

        self.policy_net = DeepQNetwork(self.n_observations, self.layer_dims[0], self.layer_dims[1], self.n_actions).to(self.device)
        self.target_net = DeepQNetwork(self.n_observations, self.layer_dims[0], self.layer_dims[1], self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)

        self.steps_done = 0
        self.record = []

    def store_transiton(self, state, action, next_state, reward, done):
        if done:
            next_state = None
        else:
            next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        self.memory.push(state, action, next_state, reward)

    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
                        math.exp(-1. * self.steps_done / self.eps_dec)
        self.epsilon = eps_threshold
        #print('epsilon', eps_threshold)
        #self.steps_done += 1
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_net.load_state_dict(target_net_state_dict)
    
    def save(self, path=None):
        torch.save(self.policy_net.state_dict(), path)
    
    def load(self, path):
        self.policy_net = DeepQNetwork(self.n_observations, self.layer_dims[0], self.layer_dims[1], self.n_actions).to(self.device)
        self.policy_net = (torch.load(path, map_location=self.device))
