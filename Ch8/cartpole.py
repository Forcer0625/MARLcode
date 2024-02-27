import gymnasium as gym
from dqn2 import DQN
import numpy as np

if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_episodes = 450
    config = {
        'batch_size':128,
        'gamma':0.99,
        'eps_start':1.0,
        'eps_end':0.00,
        'eps_dec':70,
        'tau':0.01,
        'lr':1e-3,
        'layer_dims':(64, 64),
    }
    agent = DQN(env, config)
    scores, eps_history = [], []
    

    for i in range(n_episodes):
        score = 0
        terminated = False
        truncated = False
        obs, info = env.reset()
        while not terminated and not truncated:
            action = agent.select_action(obs)
            obs_, reward, terminated, truncated, info = env.step(action.item())            
            agent.store_transiton(obs, action, obs_, reward, terminated)
            obs = obs_
            agent.learn()
            score+=reward
        agent.steps_done += 1
        scores.append(score)

        avg_score = np.mean(scores[-100:])

        print('epsiode ', i, 'score %.2f'%score,
              'avg_score %.2f'%avg_score,
              #'loss %.2f'%loss,
              'epsilon %.2f'%agent.epsilon
              )
    agent.save('Ch8\cartpole_model')
    
    score = 0
    terminated = False
    truncated = False
    env.close()
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