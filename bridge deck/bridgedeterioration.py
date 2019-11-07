# -*- coding: utf-8 -*-
import numpy as np
from utils import plot_state, processsa, time_encoder


class BridgeEnv:
    def __init__(self, tau=True):
        self.action_num = 4
        self.shape = [7, 7]
        self.time = 0
        self.state = self.reset()
        self.components = 7
        _, state_num = np.where(self.state[:, 0:6] == 1)
        self.state_num = state_num
        if tau:
            plot_state(self.state, X=7, Y=7)

    def reset(self):
        raw_s = np.zeros(self.shape,dtype=np.int32)
        self.time = 0
        raw_s[:,0] = 1 # raw state, one-hot setting; all components are in condition 1
        raw_s[:,6] = time_encoder(self.time)
        self.state = raw_s
        self.state_num = np.zeros(7,dtype=np.int32)
        return self.state

    def randomint(self,stau=False):
        s = np.random.randint(0,6,7)
        self.time = np.random.randint(0,100,1)[0]
        raw_s = np.zeros([7,7],dtype=np.int32)
        raw_s[:, 0:6] = processsa(s,6)
        raw_s[:, 6] = time_encoder(self.time)
        if stau == True:
            plot_state(raw_s,7,7)
        self.state=raw_s
        self.state_num = s
        return self.state, self.time

    def render(self):
        plot_state(self.state, 7,7)

    def costs(self,action):
        # action is a vector of length 7, denoting action conduced on each component
        total_cost = [-80, -60, -80, -60, -100, -120, -100]  # components total costs
        # wearing surface; drainage system; exterior face1; exterior face2; end portion1; middle portion; end portion2
        s_a_rate = [0.80, 0.85, 0.90, 0.95, 1.0, 1.0]  # total cost rate of 6 conditions
        a_rate = [0.0, 0.1, 0.3, 1.0]  # total cost rate of 4 actions
        risk_rate = [0.01, 0.01, 0.02, 0.03, 0.1, 0.3]
        # cost = total_cost[component_num]*s_a_rate[s]*a_rate[a]
        cost = np.zeros(7)
        s = self.state_num  # s -- condition index of
        # collapse_rate = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9]

        for component in range(self.components):
            cost[component] = total_cost[component]*s_a_rate[s[component]]*a_rate[action[component]] + \
                              total_cost[component] * risk_rate[s[component]]
                              # 100 * total_cost[component] * collapse[component]
                              # 0.03*s_a_rate[s[component]]*total_cost[component]+\
                                # maintenance costs + user costs + collapse costs
                                                                                                              
        return cost  # return sum of costs plus user_cost of the


    def step(self,action,render=False):
        self.time += 1
        done = False
        if self.time >= 100:
            done = True

        # transition possibility matrices
        trans = np.zeros(shape=[7,4,6,6])
        trans[:, 3, :, 0] = 1.0  # replace action
        trans[:, 1] = [[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.85, 0.15, 0.00, 0.00, 0.00, 0.00],
                       [0.65, 0.27, 0.08, 0.00, 0.00, 0.00],
                       [0.45, 0.30, 0.17, 0.08, 0.00, 0.00],
                       [0.30, 0.35, 0.20, 0.08, 0.07, 0.00],
                       [0.25, 0.30, 0.15, 0.17, 0.10, 0.03]]  # minor repair action
        trans[:, 2] = [[1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [1.00, 0.00, 0.00, 0.00, 0.00, 0.00],
                       [0.90, 0.08, 0.02, 0.00, 0.00, 0.00],
                       [0.70, 0.15, 0.10, 0.05, 0.00, 0.00],
                       [0.60, 0.18, 0.12, 0.07, 0.03, 0.00],
                       [0.50, 0.25, 0.15, 0.06, 0.03, 0.01]]  # major repair action
        # no-actions for all components
        trans[0,0] = [[0.81, 0.19, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.91, 0.09, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.94, 0.06, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.99, 0.01],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        trans[1,0] = [[0.91, 0.09, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.90, 0.10, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.86, 0.14, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.94, 0.06, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.93, 0.07],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        trans[2,0] = [[0.86, 0.14, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.97, 0.03, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.94, 0.06, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.89, 0.11, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.99, 0.01],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        trans[3,0] = [[0.86, 0.14, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.96, 0.04, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.95, 0.05, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.92, 0.08, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.91, 0.09],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        trans[4,0] = [[0.87, 0.13, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.95, 0.05, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.94, 0.06, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.92, 0.08, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.99, 0.01],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        trans[5,0] = [[0.87, 0.13, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.96, 0.04, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.94, 0.06, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.94, 0.06, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.99, 0.01],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        trans[6,0] = [[0.87, 0.13, 0.00, 0.00, 0.00, 0.00],
                      [0.00, 0.96, 0.04, 0.00, 0.00, 0.00],
                      [0.00, 0.00, 0.94, 0.06, 0.00, 0.00],
                      [0.00, 0.00, 0.00, 0.91, 0.09, 0.00],
                      [0.00, 0.00, 0.00, 0.00, 0.99, 0.01],
                      [0.00, 0.00, 0.00, 0.00, 0.00, 1.00]]
        # calculate costs
        reward = self.costs(action)
        # deterioration
        for component in range(self.components):  # for each component
            s = self.state_num[component]
            next_s_index = np.random.choice(6, 1, p=trans[component, action[component], s, :])
            self.state_num[component] = next_s_index

        self.state[:,0:6] = processsa(self.state_num,6)
        self.state[:,6] = time_encoder(self.time)

        if render==True:
            self.render()

        return self.state, reward, done

