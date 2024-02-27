import gymnasium as gym
from dqn2 import DQN
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_episodes = 500
    config = {
        'batch_size':128,
        'gamma':0.99,
        'eps_start':1.0,
        'eps_end':0.01,
        'eps_dec':40,
        'tau':0.01,
        'lr':1e-4,
        'layer_dims':(32, 32),
    }
    agent = DQN(env, config)
    agent.load('Ch8\cartpole_model')
    agent.steps_done = n_episodes

    score = 0
    terminated = False
    truncated = False
    #agent.epsilon = 0.0
    for i in range(100):
        env = gym.make("CartPole-v1", render_mode="human")
        score = 0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            action = agent.select_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action.item())
            obs = obs_
            score+=reward
        env.close()
    