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

class Memory():
    def __init__(self):
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()  
    
    def _zip(self):
        return zip(self.log_probs,
                self.values,
                self.rewards,
                self.dones)
    
    def __iter__(self):
        for data in self._zip():
            return data
    
    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data
    
    def __len__(self):
        return len(self.rewards)

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
        return self.model(X)
    
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

class A2C():
    def __init__(self, env, config):
        # if GPU is to be used
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = config['gamma']
        self.lr = config['lr']
        self.step_size = config['step_size']
        self.env = env
        self.layer_dims = config['layer_dims']
        self.n_actions = env.action_space.n
        self.n_observations = env.observation_space.shape[0]

        self.actor = Actor(self.n_observations, self.layer_dims[0], self.layer_dims[1], self.n_actions).to(self.device)
        self.critic = Critic(self.n_observations, self.layer_dims[0], self.layer_dims[1]).to(self.device) # sclae value for critic
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        self.memory = Memory()
        self.record = []

    def update(self, q_val):
        values = torch.stack(self.memory.values)
        q_vals = torch.zeros((len(self.memory), 1)).to(self.device)
        
        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(self.memory.reversed()):
            q_val = reward + self.gamma*q_val*(1.0-done)
            q_vals[len(self.memory)-1 - i] = torch.Tensor(q_val).to(self.device) # store values from the end to the beginning
            
        advantage = torch.Tensor(q_vals) - values
        
        critic_loss = advantage.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        actor_loss = (-torch.stack(self.memory.log_probs)*advantage.detach()).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss, critic_loss

    def learn(self, total_timesteps, render=True):
        episode_rewards = []
        avg_rewards = []
        for i in range(total_timesteps):
            terminated =False
            truncated = False
            total_reward = 0
            state, info = self.env.reset()
            steps = 0

            while not terminated and not truncated:
                probs = self.actor(torch.tensor(state).to(self.device))
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()
                
                next_state, reward, terminated, truncated, info = self.env.step(action.detach().data.cpu().numpy())
                
                total_reward += reward
                steps += 1
                self.memory.add(dist.log_prob(action), self.critic(torch.tensor(state).to(self.device)), reward, terminated)
                
                state = next_state
                
                # train if done or num steps > step_size
                if terminated or truncated or (steps % self.step_size == 0):
                    last_q_val = self.critic(torch.tensor(next_state).to(self.device)).detach().data.cpu().numpy()
                    actor_loss, critic_loss = self.update(last_q_val)
                    self.memory.clear()
                    
            episode_rewards.append(total_reward)
            avg_reward = np.mean(episode_rewards[-50:])
            avg_rewards.append(avg_reward)
            if render:
                print('epsiode ', i, 'reward %.2f'%total_reward,
                      'avg_reward %.2f'%avg_reward,
                      'actor_loss %.2f'%actor_loss,
                      'critic_loss %.2f'%critic_loss)
        
        if render:
            plt.scatter(np.arange(len(episode_rewards)), episode_rewards, s=2)
            plt.title("Total reward per episode (episodic)")
            plt.ylabel("reward")
            plt.xlabel("episode")
            plt.plot(np.arange(len(episode_rewards)), avg_rewards, color='red')
            plt.show()
        
    def select_action(self, state):
        probs = self.actor(torch.tensor(state).to(self.device))
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        return action.detach().data.cpu().numpy()
    
    def save(self, path=None):
        torch.save(self.actor.state_dict(), path+'_actor')
        torch.save(self.critic.state_dict(), path+'_critic')
    
    def load(self, path):
        self.actor = Actor(self.n_observations, self.layer_dims[0], self.layer_dims[1], self.n_actions).to(self.device)
        self.critic = Critic(self.n_observations, self.layer_dims[0], self.layer_dims[1]).to(self.device) # sclae value for critic
        self.actor.load_state_dict(torch.load(path+'_actor', map_location=self.device))
        self.critic.load_state_dict(torch.load(path+'_critic', map_location=self.device))

    
if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    task = 'play'
    config = {
        'gamma':0.99,
        'step_size':50,
        'lr':1e-3,
        'layer_dims':(64, 32),
    }
    agent = A2C(env, config)

    if task == 'train':
        agent.learn(total_timesteps=500)
        agent.save('Ch8\cartpole_a2c')
    else:
        agent.load('Ch8\cartpole_a2c')

    for i in range(100):
        env = gym.make("CartPole-v1", render_mode="human")
        score = 0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            action = agent.select_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action)
            obs = obs_
            score+=reward
        env.close()