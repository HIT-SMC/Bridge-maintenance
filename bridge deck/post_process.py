# generate manual maintenance samples
from guide_experience import generate_Q
generate_Q()

# generate DRL policy maintenance samples
import numpy as np
import tensorflow as tf
import os, time, copy
from bridgedeterioration import BridgeEnv
from policynet import CNNPolicy
from utils import getreturn, experience_buffer, processsa
num_component = 7
episodes = 1000
env = BridgeEnv()

agent = CNNPolicy()
agent.create_network(mH=128)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
ckpt = tf.train.get_checkpoint_state('./result/training results/step89300/')
saver.restore(sess, ckpt.model_checkpoint_path)

costs = []
actions = np.zeros([episodes, 100, num_component], dtype=np.int32)
states = np.zeros([episodes, 100, num_component], dtype=np.int32)
for i in range(episodes):
    _ = env.reset()
    state = env.reset()
    s = np.reshape(state,[49])
    done = False
    t, rAll = 0, 0
    while not done:
        a = sess.run(agent.predict, feed_dict={agent.scalarInput: [s]})[0]
        states[i, t, :] = copy.deepcopy(env.state_num[:num_component])
        actions[i, t, :] = copy.deepcopy(a)
        state1, reward, done = env.step(a)
        r = reward / 600
        state1 = env.state
        s1 = np.reshape(state1,[49])
        s = s1
        t += 1
        # year = time_encoder(t, 7)[np.newaxis, np.newaxis, :]
        rAll += np.sum(r)
    print(rAll)
    costs.append(rAll)

path = './result/simulation/DQN/'
if not os.path.exists(path):
    os.mkdir(path)
np.save(path + 'cost.npy', costs)
np.save(path + 'states.npy', states)
np.save(path + 'actions.npy', actions)
