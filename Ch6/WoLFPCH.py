import numpy as np
import matplotlib.pyplot as plt
import random
from copy import deepcopy
from Games import NormalFormGame, StochasticGame

class WoLFPHC():
    def __init__(self, env:StochasticGame, config) -> None:
        self.env = env
        self.n_agents = env.n_agents
        self.n_actions = env.n_actions
        self.n_states = env.n_states
        self.n_joint_actions = env.n_joint_actions
        self.epsilon = config['epsilon']
        self.lr = config['lr']
        self.l_l= config['l_l']
        self.l_w= config['l_l']
        self.gamma = config['gamma']
        self.max_steps = (int)(1/(1-self.gamma))
        self.n_episode = config['n_episode']
        self.update_epsilon = config['update_epsilon']
        self.update_lr = config['update_lr']
        self.solutions = [None for i in range(self.env.n_states)]

    def make_joint_action(self, state, epsilon):
        joint_action = []

        for i in range(self.n_agents):
            if np.random.rand() < epsilon:
                action = random.sample(range(self.n_actions[i]), 1)[0]
            else:
                action = np.random.choice([a for a in range(self.n_actions[i])],\
                                            p=self.policy[i, state]/np.sum(self.policy[i,state]))
            joint_action.append(action)

        return np.array(joint_action, dtype=int)
    
    def isWin(self, agent, state):
        policy_reward =.0
        mean_policy_reward =.0
        for a in range(self.n_actions[agent]):
            q_value = self.Q[agent, state, a]
            policy_reward += q_value*self.policy[agent, state, a]/np.sum(self.policy[agent,state])
            mean_policy_reward += q_value*self.mean_policy[agent, state, a]/np.sum(self.mean_policy[agent,state])

        if policy_reward > mean_policy_reward:
            return True
        return False
    
    def delta(self, agent, state):
        if self.isWin(agent, state):
            return self.l_w
        return self.l_l
    
    def delta_value(self, agent, state, action):
        #return np.min(self.policy[agent, state, action], (float)(self.delta(agent, state)/(self.n_actions[agent]-1)))
        left = self.policy[agent, state, action]
        right = self.delta(agent, state)/(self.n_actions[agent]-1)
        if left < right:
            return left
        return right
        
    def delta_function(self, agent, state, joint_action):
        if self.isMaxQ(agent, state, joint_action):
            sum = .0
            for a in range(self.n_actions[agent]):
                if a != joint_action[agent]:
                    sum += self.delta_value(agent, state, a)
            return sum
        return -self.delta_value(agent, state, joint_action[agent])
        
    def isMaxQ(self, agent, state, joint_action):
        if self.maxQ(agent, state) == self.Q[agent, state, joint_action[agent]]:
            return True
        return False
        
    def maxQ(self, agent, state):
        return np.max(self.Q[agent, state])
    
    def normalize_policy(self):
        for i in range(self.n_agents):
            for s in range(self.n_states):
                self.policy[i,s] = self.policy[i,s]/np.sum(self.policy[i,s])
                self.mean_policy[i,s] = self.mean_policy[i,s]/np.sum(self.mean_policy[i,s])

    def learn(self):
        # Initialize
        epsilon = self.epsilon
        # Initialize Q-values
        self.Q = np.zeros((self.n_agents, self.n_states, np.max(self.n_actions)))
        # Initialize Polcy /lazy uniform probability problem here/
        self.policy = np.full((self.n_agents, self.n_states, np.max(self.n_actions)), fill_value=(float)(1/np.max(self.n_actions)), dtype=float)
        # Initialize Average Polcy
        self.mean_policy = deepcopy(self.policy)
        # Repeat for every episode
        for episode in range(self.n_episode):
            #print(self.mean_policy[0,0])
            print(self.policy[0,0]/np.sum(self.policy[0,0]))
            #print(self.Q[0,0])
            print()
            is_done = False
            step = 0
            state = self.env.reset()
            lr = self.lr
            while (not is_done) or ((self.env.terminal_state is None) and (step < self.max_steps)):

                # Choose action use epislon-greedy
                joint_action = self.make_joint_action(state, epsilon)

                # Observe joint action, rewards, and next state
                next_state, joint_reward, is_done, info= self.env.step(joint_action)

                # Update Q-values、Average Policy and Policy for all agents
                for i in range(self.n_agents):
                    max_action_value = self.maxQ(i, state)
                    self.Q[i, state, joint_action[i]] += lr *\
                                                    (joint_reward[i]+\
                                                    self.gamma*max_action_value-\
                                                    self.Q[i, state, joint_action[i]])
                    self.mean_policy[i, state, joint_action[i]] +=  (float)(1/(episode+1))*\
                                                                (self.policy[i, state, joint_action[i]]-\
                                                                 self.mean_policy[i, state, joint_action[i]])
                    self.policy[i, state, joint_action[i]] += lr*self.delta_function(i, state, joint_action)
                
                #self.normalize_policy()
                
                # Observe next state
                state = next_state
                step += 1
                if self.update_lr is not None:
                    lr = self.update_lr(self.lr, step)
            if self.update_epsilon is not None:
                epsilon = self.update_epsilon(self.epsilon, episode)
        return self.Q
    
if __name__ == '__main__':
    sc = np.array(
        [   [0, 0],[-1, 1],[1, -1],
            [1, -1],[0, 0],[-1, 1],
            [-1, 1],[1, -1],[0, 0],]
    )
    sct = np.array(
        [[[1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0], [1.0]]]
    )
    rock_paper_scissors = StochasticGame(
        n_agents=2,
        n_states=1,
        n_actions=3,
        state_matrixs=np.array([sc]),
        state_transition_matrix=sct,
        terminal_state=[0]
    )

    # Config
    def update_lr(lr, step):
        return lr*pow(0.99999954, (int)(step))
    def update_epislon(epislon, step):
        return epislon*pow(0.999954, (int)(step))
    config={'epsilon':0.93,
            'lr':1e-2,
            'l_l':0.00081,
            'l_w':0.00064,
            'gamma':0.99,
            'n_episode':10_000,
            'update_epsilon':update_epislon,
            'update_lr':update_lr,
            }
    
    algo = WoLFPHC(env=rock_paper_scissors,
                         config=config)
    Q = algo.learn()
    print(algo.policy[1,0])

    is_done = False
    cummulative_reward = 0
    step = 0
    while True:
        step+=1
        action = np.random.choice([0,1,2],p=[1.0/3.0,1.0/3.0,1.0/3.0])
        agent1 = np.random.choice([0,1,2],p=algo.policy[1,0]/np.sum(algo.policy[1,0]))
        joint_action = [action, agent1]

        next_state, joint_reward, is_done, info = rock_paper_scissors.step(joint_action)

        cummulative_reward += joint_reward[0]
        print(cummulative_reward/step)