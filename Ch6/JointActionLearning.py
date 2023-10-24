import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import random
from copy import deepcopy
from Games import NormalFormGame, StochasticGame

class Policy:
    def __init__(self, n_agents, n_actions, distribution) -> None:
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.distribution = distribution

    def sample_action(self, agent=None):
        return (int)(np.random.randint(low=0, high=self.n_actions, size=1))
    
    def best_action(self, agent=None):
        action = np.random.choice([i for i in range(self.n_actions[0]*self.n_agents)], p=self.distribution)
        return action

    
class JointActionLearning():
    def __init__(self, env:StochasticGame, config) -> None:
        self.env = env
        self.n_agents = env.n_agents
        self.n_actions = env.n_actions
        self.n_states = env.n_states
        self.n_joint_actions = env.n_joint_actions
        self.epsilon = config['epsilon']
        self.lr = config['lr']
        self.gamma = config['gamma']
        self.max_steps = (int)(1/(1-self.gamma))
        self.n_episode = config['n_episode']
        self.update_epsilon = config['update_epsilon']
        self.solutions = [None for i in range(self.env.n_states)]

    def get_policy(self, state):
        if self.solutions[state] is None:
            self.solutions[state] = self.slove_game(state)
        return self.solutions[state]
    
    def slove_game(self, state) -> Policy:
        pass
    
    def value(self, agent, state):
        pass

    def make_joint_action(self, state):
        pass

    def learn(self):
        # Initialize Q-values
        self.Q = np.zeros((self.n_agents, self.n_states, self.n_joint_actions))
        
        # Repeat for every episode
        for episode in range(self.n_episode):
            is_done = False
            step = 0
            state = self.env.reset()
            #print("Episode {episode}:".format(episode=episode))
            while (not is_done) and ((self.env.terminal_state is None) and (step < self.max_steps)):
                #print("---Step {step}".format(step=step))
                # Choose action
                joint_action = self.make_joint_action(state)

                # Observe joint action, rewards, and next state
                next_state, joint_reward, is_done, info= self.env.step(joint_action)
                joint_action = self.env.flatten_joint_action(joint_action)
                
                # Update Q-values for all agents
                for j in range(self.n_agents):
                    self.Q[j, state, joint_action] += self.lr *\
                                                    (joint_reward[j]+\
                                                    self.gamma*self.value(j, next_state)-\
                                                    self.Q[j, state, joint_action])
                # Observe next state
                state = next_state
                step += 1
            if self.update_epsilon is not None:
                self.epsilon = self.update_epsilon(episode)
        return self.Q
    
    def evaluate(self, optimal:Policy, rand=False):
        n_eval_epi = 100
        colors = ['aqua', 'blue', 'green', 'yellow', 'darkgreen', 'lightyellow', 'orange']
        # Correlated Q-learning
        corrq = []
        # joint action value: using maximum of sum of agents' reward
        joint_Q = np.zeros(shape=(self.n_states, self.n_joint_actions))
        for s in range(self.n_states):
            for a in range(self.n_joint_actions):
                joint_Q[s,a] = np.sum(self.Q[:,s,a])

        for episode in range(n_eval_epi):
            is_done = False
            step = 0
            state = self.env.reset()
            e_reward = 0.0
            while (not is_done) and ((self.env.terminal_state is None) and (step < self.max_steps)):
                #print("---Step {step}".format(step=step))
                # Choose best action
                if rand and np.random.rand() < self.epsilon:
                    joint_action = optimal[state].sample_action()
                    joint_action = self.env.unflatten_joint_action(joint_action)
                else:
                    joint_action = (int)(np.argmax(joint_Q[state]))
                
                # Observe joint action, rewards, and next state
                next_state, joint_reward, is_done, info= self.env.step(joint_action)
                
                # Observe next state
                state = next_state
                step += 1

                # Calculate reward
                e_reward += np.sum(joint_reward)
            corrq.append(e_reward)

        # Optimal Policy
        opt = []
        for episode in range(n_eval_epi):
            is_done = False
            step = 0
            state = self.env.reset()
            e_reward = 0.0
            while (not is_done) and ((self.env.terminal_state is None) and (step < self.max_steps)):
                #print("---Step {step}".format(step=step))
                # Choose best action
                joint_action = optimal[state].best_action()
                joint_action = self.env.unflatten_joint_action(joint_action)

                # Observe joint action, rewards, and next state
                next_state, joint_reward, is_done, info= self.env.step(joint_action)
                
                # Observe next state
                state = next_state
                step += 1

                # Calculate reward
                e_reward += np.sum(joint_reward)
            opt.append(e_reward)
            
        plt.plot(corrq, color=colors[0])
        plt.plot(opt, color=colors[1])
        plt.plot([0], color=colors[2])
        plt.plot([0], color=colors[3])
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.legend(['Correlated Q-learning', 'Optimal Polciy', 'Average of optimal policy:{value}'.format(value=np.mean(opt)), 'Average of CorrQ:{value}'.format(value=np.mean(corrq))], loc='best')
        #plt.savefig('Ch6\Value Iteration MSE Convergence.png')
        plt.show()

class CorrelatedQlearning(JointActionLearning):
    def __init__(self, env:StochasticGame, config=
                 {'epsilon':0.7,
                  'lr':1e-3,
                  'gamma':0.9,
                  'n_episode':1000,
                  'update_epsilon':None,
                  }) -> None:
        super().__init__(env, config)
    
    def slove_game(self, state) -> Policy:
        state_matrix = self.env.state_matrix(state)
        n_joint_actions = self.env.n_joint_actions
        # Linear Programming
        # 1. Objective Function: Maximize sum of all agents' reward
        obj = [-np.sum(state_matrix[a]) for a in range(self.env.n_joint_actions)]
        # 2. Constraints
        lhs_eq = [[1 for a in range(n_joint_actions)]]
        rhs_eq = [1.0]

        lhs_ineq = [[[0.0 for a in range(n_joint_actions)] for i in range(self.n_agents)] for k in range((self.env.n_actions[0]-1))]
        for agent in range(self.env.n_agents):
            for ja in range(n_joint_actions):
                joint_action = self.env.unflatten_joint_action(ja)
                other_action_count = 0
                for a in range(self.env.n_actions[agent]):
                    other_joint_action = deepcopy(joint_action)
                    other_joint_action[agent] = a
                    oja = self.env.flatten_joint_action(other_joint_action)
                    if ja != oja:
                        lhs_ineq[other_action_count][agent][ja] = (float)(state_matrix[oja][agent]-state_matrix[ja][agent])
                        other_action_count+=1


        lhs_ineq = np.array(lhs_ineq)
        lhs_ineq = np.reshape(lhs_ineq, newshape=(-1, n_joint_actions))
        lhs_ineq = lhs_ineq.tolist()

        rhs_ineq = [0.0 for i in range(self.env.n_agents*(self.env.n_actions[0]-1))]

        bound = [(0.0, 1.0) for a in range(n_joint_actions)]
        # 3. Optimize
        opt = linprog(c=obj,
                      A_ub=lhs_ineq, b_ub=rhs_ineq,
                      A_eq=lhs_eq, b_eq=rhs_eq,
                      bounds=bound)
        
        joint_actions_distribution = opt.x
        return Policy(n_agents=self.n_agents, n_actions=self.n_actions, distribution=joint_actions_distribution)
    
    def value(self, agent, state):
        state_matrix = self.env.state_matrix(state)
        max = np.sum(state_matrix[0], dtype=float)
        for a in range(self.env.n_joint_actions-1):
            sum = np.sum(state_matrix[a+1], dtype=float)
            if sum > max:
                max = sum    
        return max

    def make_joint_action(self, state):
        joint_action = []
        policy = self.get_policy(state)
        if np.random.rand() < self.epsilon:
            for i in range(self.n_agents):
                action = policy.sample_action(i)
                joint_action.append(action)
        else:
            joint_action = policy.best_action()
            joint_action = self.env.unflatten_joint_action(joint_action)

        return np.array(joint_action, dtype=int)

if __name__ == '__main__':
    sc = np.array(
        [   [0, 0],[7, 2],
            [2, 7],[6, 6]]
    )
    sct = np.array(
        [[[1.0], [1.0], [1.0], [1.0]]]
    )
    chicken_game = StochasticGame(
        n_agents=2,
        n_states=1,
        n_actions=2,
        state_matrixs=np.array([sc]),
        state_transition_matrix=sct,
    )
    s1 = np.array(
        [   [1, 0],[1, 0],
            [0, 3],[0, 3]]
    )
    s2 = np.array(
        [   [3, 1],[0, 0],
            [3, 1],[0, 0]]
    )
    st = np.array([
            [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])
    def nosde_init():
        return random.randint(0, 1)
    nosde = StochasticGame(
        n_agents=2,
        n_states=2,
        n_actions=2,
        state_matrixs=np.array([s1, s2]),
        state_transition_matrix=st,
        #initial_state=nosde_init,
    )
    
    algo = CorrelatedQlearning(env=chicken_game)
    joint_action_value = algo.learn()
    algo.evaluate([algo.slove_game(state=0)], rand=False)

    algo = CorrelatedQlearning(env=nosde)
    joint_action_value = algo.learn()
    best_distribution = [1/3*7/12, 1/3*5/12, 2/3*7/12, 2/3*5/12]
    algo.evaluate([Policy(n_agents=2, n_actions=[2, 2], distribution=best_distribution),
                   Policy(n_agents=2, n_actions=[2, 2], distribution=best_distribution)], rand=False)