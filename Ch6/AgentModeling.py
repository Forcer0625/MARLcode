import numpy as np
import matplotlib.pyplot as plt
import random
from Games import NormalFormGame, StochasticGame
from JointActionLearning import JointActionLearning

class AgentModel():
    def __init__(self, n_agents, n_states, n_actions):
        self.policy = np.zeros((n_agents, n_states, n_actions), dtype=int)

    def update(self, state, joint_action):
        for i in range(len(joint_action)):
            self.policy[i][state][joint_action[i]] += 1

    def prob_except_agent(self, agent, state, joint_action):
        prob = 1.0
        for i in range(len(joint_action)):
            if agent != i:
                if np.sum(self.policy[i,state]) == 0:
                    prob *= 1.0/len(self.policy[i,state])
                else:
                    prob *= self.policy[i][state][joint_action[i]]/np.sum(self.policy[i,state])
        return  prob

class AgentModeling():
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
        self.update_lr = config['update_lr']
        self.solutions = [None for i in range(self.env.n_states)]
    
    def action_value(self, agent, state, agent_model: AgentModel):
        action_values = np.zeros(self.n_actions[agent])

        for a in range(self.n_actions[agent]):
            joint_action = np.zeros(self.n_agents, dtype=int)
            joint_action[agent] = a
            sum = .0
            for j in range(self.n_agents):
                if j!=agent:
                    for oa in range(self.n_actions[j]):
                        joint_action[j] = oa
                        sum += self.Q[agent,state,self.env.flatten_joint_action(joint_action)]*\
                                agent_model.prob_except_agent(agent,state,joint_action)
            action_values[a] = sum

        return action_values.max(), np.argmax(action_values)

    def make_joint_action(self, state, epsilon, agent_model):
        joint_action = []

        for i in range(self.n_agents):
            if np.random.rand() < epsilon:
                action = random.sample(range(self.n_actions[i]), 1)[0]
            else:
                value, action = self.action_value(i, state, agent_model)
            joint_action.append(action)

        return np.array(joint_action, dtype=int)

    def learn(self):
        # Initialize
        epsilon = self.epsilon
        # Initialize Q-values
        self.Q = np.zeros((self.n_agents, self.n_states, self.n_joint_actions))
        # Initialize Agent Models
        agent_model = AgentModel(self.n_agents, self.n_states, np.max(self.n_actions))
        print("Rock|Paper|Scissors")
        # Repeat for every episode
        for episode in range(self.n_episode):
            is_done = False
            step = 0
            state = self.env.reset()
            lr = self.lr
            while (not is_done) or ((self.env.terminal_state is None) and (step < self.max_steps)):

                # Choose action use epislon-greedy
                joint_action = self.make_joint_action(state, epsilon, agent_model)

                # Observe joint action, rewards, and next state
                next_state, joint_reward, is_done, info= self.env.step(joint_action)
                # Update Agent Model
                agent_model.update(state, joint_action)
                joint_action = self.env.flatten_joint_action(joint_action)

                # Update Q-values for all agents
                for i in range(self.n_agents):
                    max_action_value, best_action = self.action_value(i, next_state, agent_model)
                    self.Q[i, state, joint_action] += lr *\
                                                    (joint_reward[i]+\
                                                    self.gamma*max_action_value-\
                                                    self.Q[i, state, joint_action])
                
                # Observe next state
                state = next_state
                step += 1
                if self.update_lr is not None:
                    lr = self.update_lr(self.lr, step)
            q_sum = np.sum(self.Q[0,0])
            print("{r}|{p}|{s}\t\t|{rt}|{pt}|{st}".format(r=round(np.sum(self.Q[0,0,0:3])/q_sum,2),
                                       p=round(np.sum(self.Q[0,0,3:6])/q_sum,2),
                                       s=round(np.sum(self.Q[0,0,6:9])/q_sum,2),
                                       rt=agent_model.policy[1, 0, 0],
                                       pt=agent_model.policy[1, 0, 1],
                                       st=agent_model.policy[1, 0, 2]))
            if self.update_epsilon is not None:
                epsilon = self.update_epsilon(self.epsilon, episode)
                print(epsilon)
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
    config={'epsilon':0.9,
            'lr':1e-1,
            'gamma':.99,
            'n_episode':30000,
            'update_epsilon':update_epislon,
            'update_lr':update_lr,
            }
    
    algo = AgentModeling(env=rock_paper_scissors,
                         config=config)
    Q = algo.learn()

    is_done = False
    
    while True:
        print("0:rock\t1:paper\t2:scissors")
        action = int(input('your action:'))
        agent1 = np.argmax([round(np.sum(Q[0,0,0:3]),2), round(np.sum(Q[0,0,3:6]),2), round(np.sum(Q[0,0,6:9]),2)])
        joint_action = [agent1, action]
        
        print(joint_action)

        next_state, joint_reward, is_done, info = rock_paper_scissors.step(joint_action)

        if joint_reward[1] > 0:
            print("You Win")
        elif joint_reward[1] < 0:
            print("You Lose")
        else:
            print("It's a Tie")
        print("")
