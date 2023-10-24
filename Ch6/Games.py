import numpy as np
import random

class NormalFormGame:
    def __init__(self, n_agents:int, n_actions:tuple|int, state_matrix) -> None:
        self.n_agents = n_agents
        if type(n_actions)==tuple:
            self.n_actions = n_actions
        else:
            self.n_actions = tuple(n_actions for i in range(n_agents))
        self.n_joint_actions = 1
        for actions in self.n_actions:
            self.n_joint_actions*=actions
        self.state_matrix = np.reshape(state_matrix, (-1, n_agents))

    def flatten_joint_action(self, joint_action):
        if type(joint_action) is int:
            return joint_action
        index = 0
        for i in range(self.n_agents):
            index += (joint_action[i]*pow(self.n_actions[0],self.n_agents-i-1))
        return index
    
    def unflatten_joint_action(self, joint_action):
        actions = []
        for i in range(self.n_agents):
            i_action = joint_action//pow(self.n_actions[0], self.n_agents-i-1)
            joint_action -= (i_action*pow(self.n_actions[0],self.n_agents-i-1))
            actions.append(i_action)
        return np.array(actions, dtype=int)
    
    def step(self, joint_action:np.ndarray|tuple):
        return self.joint_reward(joint_action)
    
    def joint_reward(self, joint_action:np.ndarray|tuple):
        return self.state_matrix[self.flatten_joint_action(joint_action)]
    
    def reward(self, joint_action:np.ndarray|tuple, agent=None):
        if agent is None:
            return self.joint_reward(joint_action)
        return self.joint_reward(joint_action)[agent]
    
    

class StochasticGame(NormalFormGame):
    def __init__(self,
                 n_agents:int,
                 n_states:int,
                 n_actions:tuple|int,
                 state_matrixs,
                 state_transition_matrix,
                 initial_state=None,
                 terminal_state=None,
                 ) -> None:
        self.n_agents = n_agents
        self.n_states = n_states
        if type(n_actions)==tuple:
            self.n_actions = n_actions
        else:
            self.n_actions = tuple(n_actions for i in range(n_agents))
        self.n_joint_actions = 1
        for actions in self.n_actions:
            self.n_joint_actions*=actions
        self.state_matrixs = np.reshape(state_matrixs, (n_states, -1, n_agents))
        '''[state][joint_action][agent]'''
        self.state_transition_matrix = np.reshape(state_transition_matrix, (n_states, self.n_joint_actions, n_states))
        ''' [state][joint_action][next_state]'''
        self.initial_state = initial_state
        self.terminal_state = terminal_state
        self.state = self.reset()

    def reset(self, start_from=None):
        if self.initial_state is None:
            if start_from is None:
                return 0
            else:
                return random.randint(0, self.n_states-1)
        return self.initial_state()
    
    def state_matrix(self, state=None):
        if state is None:
            return self.state_matrixs[self.state]
        return self.state_matrixs[state]
    
    def joint_reward(self, joint_action: np.ndarray | tuple, state=None):
        index = self.flatten_joint_action(joint_action)
        if state is None:
            return self.state_matrixs[self.state][index]
        return self.state_matrixs[state][index]
        
    def step(self, joint_action:np.ndarray|tuple):
        joint_reward = self.joint_reward(joint_action)
        index = self.flatten_joint_action(joint_action)
        next_state = np.random.choice([i for i in range(self.n_states)], p=self.state_transition_matrix[self.state, index])
        self.state = next_state
        is_terminated = False
        if self.terminal_state is not None:
            is_terminated = next_state in self.terminal_state
        
        return next_state, joint_reward, is_terminated, None


# if __name__ == "__main__":
#     s1 = np.array(
#         [   [1, -1],[0, 0],
#             [2, -2],[-1, 1]]
#     )
#     s2 = np.array(
#         [   [3, -3],[0, 0],
#             [-2, 2],[-3, 3]]
#     )
#     s3 = np.array(
#         [   [0, 0],[0, 0],
#             [1, -1],[-1, 1]]
#     )
#     st = np.array([
#             [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
#             [[0.0, 1.0, 0.0], [0.8, 0.0, 0.2], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
#             [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
#         ], ndmin=3)
#     joint_action = np.array([1, 1])
#     test = StochasticGame(
#         n_agents=2,
#         n_states=3,
#         n_actions=2,
#         state_matrixs=np.array([s1, s2, s3]),
#         state_transition_matrix=st,
#     )
#     print(test.reward(joint_action=joint_action))
#     test.step(joint_action=joint_action)