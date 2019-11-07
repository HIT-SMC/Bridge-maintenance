# -*- coding: utf-8 -*-
import numpy as np
import scipy.io as sio

# total costs of components are predefined in the .mat file
costs = sio.loadmat('./total_cost.mat')['all']


class BridgeEnv:
    def __init__(self, num_components):
        self.action_num = 4  # action dimension
        self.time = 0
        self.components = num_components  # 168 cables + 89 box girder sections + 2 towers + 4 piers
        self.state = self.reset()  # state dimension: 263 component-dimension + 1 time-dimension
        trans = np.zeros(shape=[4,4,6,6])  # 4 type of components; 4 action level; 6 states*6states
        # replace action
        trans[:, 3, :, 0] = 1.0
        # minor repair action
        trans[:, 1] = [[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.80, 0.20, 0.00, 0.00, 0.00, 0.00],
                       [0.40, 0.50, 0.10, 0.00, 0.00, 0.00],
                       [0.20, 0.30, 0.40, 0.10, 0.00, 0.00],
                       [0.10, 0.20, 0.30, 0.32, 0.08, 0.00],
                       [0.05, 0.10, 0.15, 0.32, 0.35, 0.03]]
        # major repair action
        trans[:, 2] = [[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.90, 0.08, 0.02, 0.00, 0.00, 0.00],
                       [0.70, 0.15, 0.10, 0.05, 0.00, 0.00],
                       [0.60, 0.18, 0.12, 0.07, 0.03, 0.00],
                       [0.50, 0.25, 0.15, 0.06, 0.03, 0.01]]
        # no-actions for all components
        # 0 is for the cables
        trans[0,0] = [[0.92, 0.08, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.91, 0.09, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.91, 0.09, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.90, 0.10, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.90, 0.10],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        # 1 is for the box girder
        trans[1,0] = [[0.85, 0.15, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.84, 0.16, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.81, 0.19, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.78, 0.22, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.75, 0.25],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        # 2 is for the bridge tower
        trans[2,0] = [[0.96, 0.04, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.94, 0.06, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.92, 0.08, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.90, 0.10, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.89, 0.11],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        trans[3,0] = [[0.95, 0.05, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.93, 0.07, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.91, 0.09, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.90, 0.10, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.88, 0.12],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        self.trans = trans

    def reset(self):
        self.state = np.zeros([self.components + 1], dtype=np.int32)
        self.time = 0
        return self.state

    def randomint(self):
        s = np.random.randint(0, 6, self.components)  # set random initialization states for pre-train
        self.time = np.random.randint(0, 100, 1)[0]
        self.state[0:self.components] = s
        self.state[-1] = self.time
        return self.state, self.time

    def costs(self,action):
        total_cost = -costs  # components total costs
        s_a_rate = [0.80, 0.85, 0.90, 0.95, 1.0, 1.0]  # total cost rate of 6 conditions
        a_rate = [0.0, 0.1, 0.3, 1.0]  # total cost rate of 4 actions
        user_cost = [0.01, 0.01, 0.02, 0.03, 0.05, 0.2]
        cost = np.zeros(self.components)
        s = self.state  # component statef.components)
        s = self.state  # c
        # collapse_rate = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

        for component in range(self.components):
            cost[component] = total_cost[component]*s_a_rate[s[component]]*a_rate[action[component]] + \
                              total_cost[component] * user_cost[s[component]]
                              # 100 * total_cost[component] * collapse[component]
                              # 0.03*s_a_rate[s[component]]*total_cost[component]+\
                                # maintenance costs + user costs + collapse costs
                                                                                                              
        return cost  # return sum of costs plus user_cost of the

    def step(self,action):
        self.time += 1
        done = False
        if self.time >= 100:
            done = True

        # transition possibility matrices
        # calculate costs
        reward = self.costs(action)
        # deterioration
        for component in range(self.components):  # for each component
            s = self.state[component]
            if component < 168:
                type = 0
            elif component < 257:
                type = 1
            elif component < 259:
                type = 2
            elif component <263:
                type = 3
            next_s_index = np.random.choice(6, 1, p=self.trans[type, action[component], s, :])
            self.state[component] = next_s_index
        self.state[-1] = self.time

        return self.state, reward, done