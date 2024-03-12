import math
import random
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gymnasium as gym

# helper function to convert numpy arrays to tensors
def t(x): return torch.from_numpy(x).float()
def clip(a, b, c):
    if a < b:
        return b
    if a > c:
        return c
    return a

class Memory():
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class Actor(nn.Module):
    def __init__(self, n_observations, layer1_dims, layer2_dims, n_actions):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(n_observations, layer1_dims),
            nn.Tanh(),
            nn.Linear(layer1_dims, layer2_dims),
            nn.Tanh(),
            nn.Linear(layer2_dims, n_actions),
            nn.Softmax() # action probability
        )
    
    def forward(self, X):
        dist = self.model(X)
        dist = torch.distributions.Categorical(dist)
        return dist
    
class Critic(nn.Module):
    def __init__(self, n_observations, layer1_dims, layer2_dims):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations, layer1_dims)
        self.layer2 = nn.Linear(layer1_dims, layer2_dims)
        self.layer3 = nn.Linear(layer2_dims, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)

class PPO():
    def __init__(self, env, config):
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config['gamma']
        self.gae_lambda = config['gae_lambda']
        self.lr = config['lr']
        self.step_size = config['step_size']
        self.clip_size = config['clip_size']
        self.epoch_size = config['epoch_size']
        self.env = env
        self.layer_dims = config['layer_dims']
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.shape[0]

        self.actor = Actor(self.n_observations, self.layer_dims[0], self.layer_dims[1], self.n_actions).to(self.device)
        self.critic = Critic(self.n_observations, self.layer_dims[0], self.layer_dims[1]).to(self.device) # sclae value for critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.memory = Memory(config['mem_size'])
        self.record = []

    def store(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def update(self):
        for _ in range(self.epoch_size):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.device)

            values = torch.tensor(values).to(self.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float).to(self.device)
                old_probs = torch.tensor(old_prob_arr[batch]).to(self.device)
                actions = torch.tensor(action_arr[batch]).to(self.device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                #prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.clip_size,
                        1+self.clip_size)*advantage[batch]
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

        self.memory.clear_memory()

    def learn(self, total_timesteps, render=True):
        episode_rewards = []
        avg_rewards = []
        for i in range(total_timesteps):
            terminated =False
            truncated = False
            total_reward = 0
            observation, info = self.env.reset()
            steps = 0

            while not terminated and not truncated:
                action, prob, val = self.select_action(observation)
                next_observation, reward, terminated, truncated, info = self.env.step(action)
                
                total_reward += reward
                steps += 1
                self.store(observation, action, prob, val, reward, terminated)
                
                if steps % self.step_size == 0:
                    self.update()
                
                observation = next_observation
                    
            episode_rewards.append(total_reward)
            avg_reward = np.mean(episode_rewards[-50:])
            avg_rewards.append(avg_reward)
            if render:
                print('epsiode ', i, 'reward %.2f'%total_reward,
                      'avg_reward %.2f'%avg_reward,)
        
        if render:
            plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
            plt.title("Total reward per episode (episodic)")
            plt.ylabel("reward")
            plt.xlabel("episode")
            plt.plot(np.arange(len(episode_rewards)), avg_rewards, color='red')
            plt.show()
        
    def select_action(self, observation):
        state = torch.tensor([observation], dtype=torch.float).to(self.device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = torch.squeeze(dist.log_prob(action)).item()
        action = torch.squeeze(action).item()
        value = torch.squeeze(value).item()

        return action, probs, value
    
    def save(self, path=None):
        torch.save(self.actor.state_dict(), path+'_actor')
        torch.save(self.critic.state_dict(), path+'_critic')
    
    def load(self, path):
        self.actor = Actor(self.n_observations, self.layer_dims[0], self.layer_dims[1], self.n_actions).to(self.device)
        self.critic = Critic(self.n_observations, self.layer_dims[0], self.layer_dims[1]).to(self.device) # sclae value for critic
        self.actor.load_state_dict(torch.load(path+'_actor', map_location=self.device))
        self.critic.load_state_dict(torch.load(path+'_critic', map_location=self.device))

    
if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    task = 'train'
    config = {
        'gamma':0.99,
        'gae_lambda':0.95,
        'step_size':5,
        'clip_size':0.2,
        'epoch_size':4,
        'mem_size':64,
        'lr':3e-4,
        'layer_dims':(64, 32),
    }
    agent = PPO(env, config)

    if task == 'play':
        agent.learn(total_timesteps=1000, render=True)
        agent.save('Ch8\LunarLander_ppo')
    else:
        agent.load('Ch8\LunarLander_ppo')

    for i in range(100):
        env = gym.make("LunarLander-v2", render_mode="human")
        score = 0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            action, prob, val = agent.select_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            obs = obs_
            score+=reward
        env.close()