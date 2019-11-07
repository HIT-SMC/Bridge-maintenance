# **************************************************************************************************************************
"costs & loss figures"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
font = {"fontname": 'Times New Roman', "fontsize": 16}
font1 = {"fontname": 'Times New Roman', "fontsize": 12}
costs = np.load('./result/training results/step35000/costs.npy')
loss = np.load('./result/training results/step35000/DQNloss.npy')
A = np.reshape(costs[101:], [349,100])
B = np.reshape(loss, [349,100])
l1, = plt.plot(np.arange(1,350), -np.log10(-np.mean(A, 1)),linewidth=2, label='Total costs')
l2, = plt.plot(np.arange(1,350), np.log10(np.mean(B, 1)),linewidth=2, label= 'DQN loss')
plt.xlabel('Training steps', **font)
plt.ylabel('Results', **font)
plt.xlim([-50, 950]);plt.grid()
plt.xticks(np.arange(0,901, 300), ['0', '30000', '60000', '90000'], **font1)
plt.yticks(np.arange(-1,5,1), ['-10', '0', '10', '100', '1000', '10000'], **font1)
plt.legend([l1, l2], ['Total costs', 'DQN loss'], loc = 'upper right', fontsize=16)
plt.savefig('./figures/cost&loss.pdf')
plt.savefig('./figures/cost&loss.eps')


# plot paper figures
"# plot encoded_state fig-**"
import numpy as np
from bridgedeterioration import BridgeEnv
from utils import *
import matplotlib.pyplot as plt
num_component = 263
env = BridgeEnv(num_component)
env.randomint()
s = env.state[:num_component]
encode_s = processsa(s, 6)
encode_s = np.reshape(encode_s, [1, num_component*6])
year = time_encoder(env.time, 22)[np.newaxis, :]
SS = np.hstack([encode_s, year])
S = np.reshape(SS, [40,40])
plot_state(S, 40, 40)
plt.savefig('./figures/state_263.pdf')
plt.savefig('./figures/state_263.eps')

# comparison among DRL c1c2c3;  Table 3  *******************************************************************************
from guide_experience import generate_Q
generate_Q()

# *********************** DRL *****************************************************
import numpy as np
import os, time, copy, pickle
from bridgedeterioration import BridgeEnv
from utils import getreturn, experience_buffer, time_encoder
from policy import CNNagent
import tensorflow as tf

num_component = 263
env = BridgeEnv(num_component)
episodes = 1000
gamma = 0.95

agent = CNNagent()
saver = tf.train.Saver(max_to_keep=100)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state('./result/training results/step110000/')
saver.restore(sess, ckpt.model_checkpoint_path)

costs = []
Q_dict = {}
actions = np.zeros([episodes, 100, num_component], dtype=np.int32)
states = np.zeros([episodes, 100, num_component], dtype=np.int32)
for i in range(episodes):
    _ = env.reset()
    s = env.state[:num_component]
    done = False
    t, rAll = 0, 0
    year = time_encoder(env.time, 22)[np.newaxis, :]

    while not done:
        a = sess.run(agent.actions, feed_dict={agent.s: s[np.newaxis,:], agent.bin_year: year})[0]
        states[i, t, :] = copy.deepcopy(s)
        actions[i, t, :] = copy.deepcopy(a)
        state1, reward, done = env.step(a)
        r = reward / 600
        s1 = env.state[:num_component]
        s = s1
        t += 1
        year = time_encoder(t, 22)[np.newaxis, :]
        rAll += np.sum(r)
    print(rAll)
    costs.append(rAll)

path = './result/simulation/DQN/'
if not os.path.exists(path):
    os.mkdir(path)
np.save(path + 'cost.npy', costs)
np.save(path + 'states.npy', states)
np.save(path + 'actions.npy', actions)