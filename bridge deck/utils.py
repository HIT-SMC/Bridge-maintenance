import numpy as np
import matplotlib.pyplot as plt
import random


# help functions
def plot_state(state, X, Y):
    s = state.copy()
    a = s == 1
    b = s == 0
    s[a] = 0
    s[b] = 1
    plt.imshow(s, interpolation='nearest', cmap=plt.cm.gray)
    font = {"fontname": 'Times New Roman', "fontsize": 16}
    for i in range(X):
        for j in range(Y):
            plt.plot([j - 0.5, j - 0.5], [-0.5, X - 0.5], 'k')
        plt.plot([-0.5, Y - 0.5], [i - 0.5, i - 0.5], 'k')
        plt.xlabel('condition', **font)
        plt.ylabel('components', **font)
        plt.xticks(fontname='times new roman')
        plt.yticks(fontname='times new roman')


def time_encoder(time):
    coder = np.zeros(7,dtype=np.int32)
    for i in range(7):
        coder[i] = divmod(time,2**(6-i))[0]
        time -= coder[i]*2**(6-i)
    return coder


def time_decoder(time_code):
    time = 0
    for i in range(7):
        time += time_code[i]*2**(6-i)
    return time


def num_encoder(inputs,base):
    num = 0
    assert inputs.shape == (7,)
    for i in range(inputs.size):
        num += inputs[i]*base**(6-i)
    return num


def num_decoder(inputs,base):
    outs = np.zeros(7)
    for i in range(7):
        outs[i] = int(divmod(inputs,base**(6-i))[0])
        inputs -= outs[i]*base**(6-i)
    return np.int32(outs)


def getreturn(reward_list, gamma):  # calculate return of a trajectory
    # gamma is the discount factor
    a,b = reward_list.shape
    G = np.zeros([a,b])
    for i in range(a):
        for j in range(i,a):
            G[i] += gamma**(j-i)*reward_list[j,:]
    return G


def processsa(input,base):
    # 6 for states,  4 for actions
    num = input.size
    out = np.zeros([num,base],dtype=np.int32)
    for i,a in enumerate(input):
        out[i,:] = np.eye(1,base,a)
    return out


class experience_buffer():
    def __init__(self, buffer_size):  # buffer size = 50000
        self.buffer = []
        self.buffer_size = buffer_size
    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, len(self.buffer[0])])

