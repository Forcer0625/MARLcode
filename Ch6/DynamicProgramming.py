import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from Games import NormalFormGame, StochasticGame

def mean_square_error(a, b):
        return np.square(np.subtract(a, b)).mean()

class ValueIteration:
    def __init__(self, env: StochasticGame, gamma=0.99, max_steps=None):
        self.env = env
        self.gamma = gamma
        self.converge_bound = .05
        self.value = np.zeros(shape=(env.n_agents, env.n_states))
        self.max_steps = max_steps

    def value_iteration(self):
        # 1. Initialize Vi(s)
        self.value = np.zeros(shape=(env.n_agents, env.n_states))
        matrix = np.zeros(shape=(self.env.n_agents, self.env.n_states, self.env.n_joint_actions))
        record = []
        for i in range(self.env.n_agents):
            record.append([])
        if self.max_steps is None:
            prev_value = deepcopy(self.value)
        step = 0
        flag = True

        # 2. Repeat 3.~4. until converged or reach max steps
        while flag:
            # 3. Compute matrix
            for i in range(self.env.n_agents):
                for s in range(self.env.n_states):
                    for a in range(self.env.n_joint_actions):
                        self.compute_matrix(matrix, i ,s ,a)

            # 4. Update Vi(s)
            for i in range(self.env.n_agents):
                for s in range(self.env.n_states):
                    self.update(matrix, i, s)
                mse = mean_square_error(prev_value, self.value)
                record[i].append(mse)
                if mse <= self.converge_bound-0.04995:
                    flag = False
                elif self.max_steps is not None and step >= self.max_steps:
                    flag = False
            prev_value = deepcopy(self.value)

        return record

    def update(self, mat, agent, state):
        self.value[agent, state] = np.max(mat[agent, state])

    def compute_matrix(self, mat, agent, state, joint_action):
        mat[agent, state, joint_action] = 0
        for next_s in range(self.env.n_states):
            mat[agent, state, joint_action]+=\
                self.env.state_transition_matrix[state, joint_action, next_s]*\
                (self.env.state_matrixs[state, joint_action, agent]+self.gamma*self.value[agent, next_s])

    def non_repeated_normal_from_game_minimax(self):
        pass

if __name__ == '__main__':
    # 1. Create stochastic game as an environment
    s1 = np.array(
        [   [1, -1],[0, 0],
            [2, -2],[-1, 1]]
    )
    s2 = np.array(
        [   [3, -3],[0, 0],
            [-2, 2],[-3, 3]]
    )
    s3 = np.array(
        [   [0, 0],[0, 0],
            [1, -1],[-1, 1]]
    )
    state_transition_matrix = np.array([
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
            [[0.0, 1.0, 0.0], [0.8, 0.0, 0.2], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ], ndmin=3)
    env = StochasticGame(
        n_agents=2,
        n_states=3,
        n_actions=2,
        state_matrixs=np.array([s1, s2, s3]),
        state_transition_matrix=state_transition_matrix,
    )
    algo = ValueIteration(env=env, gamma=0.99)
    record = algo.value_iteration()
    #for i in range(env.n_agents):
    plt.plot(record[0], color='aqua')
    plt.plot(record[1], color='blue')
    plt.plot([0.05 for i in range(max(len(record[0]),len(record[1])))], 'r:')
    plt.xlabel('Steps')
    plt.ylabel('Mean Square Error')
    plt.legend(['Agent 1', 'Agent 2', 'Convergence Boundary:{value}'.format(value=algo.converge_bound)], loc='best')
    print(algo.value)
    plt.savefig('Ch6\Value Iteration MSE Convergence.png')
    plt.show()
    