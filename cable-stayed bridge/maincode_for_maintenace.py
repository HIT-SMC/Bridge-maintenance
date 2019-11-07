# this is the main code for maitenance policy searching by RL
import numpy as np
import tensorflow as tf
import os, time, copy, pickle
from bridgedeterioration import BridgeEnv
from utils import getreturn, experience_buffer, time_encoder
from guide_experience import generate_Q
from policy import CNNagent

if not os.path.exists('./result/'):
    os.mkdir('./result/')
    os.mkdir('./result/training results/')

# load experiences
if not os.path.exists('./result/Qvalue.pickle'):
    generate_Q()
    with open('./result/Qvalue.pickle', 'rb') as saved:
        Q_dict = pickle.load(saved)
else:
    with open('./result/Qvalue.pickle', 'rb') as saved:
        Q_dict = pickle.load(saved)
tf.reset_default_graph()


# hyper parameters
num_component = 263
env = BridgeEnv(num_component)
batch_size = int(1e3)
startE, endE = 1, 0.1
e_step = (startE-endE)/1e4
e = startE
capacity = int(1e5)
gamma = 0.95
agent = CNNagent()
saver = tf.train.Saver(max_to_keep=100)
costs, DQNloss = [], []
Buffer = experience_buffer(capacity)
qloss = []
update_freq = 1
num_episodes = int(1e6)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

load_model = False
if load_model:
    ckpt = tf.train.get_checkpoint_state('./result/training results/step110000/')
    saver.restore(sess, ckpt.model_checkpoint_path)

start = time.time()
for i_episode in range(num_episodes+1):
    state = env.reset()
    s = env.state[np.newaxis,:num_component]
    year = time_encoder(env.time, 22)[np.newaxis, :]
    done = False
    t, rAll = 0, 0
    states = np.zeros([100, num_component], dtype=np.int32)
    states_ = np.zeros([100, num_component], dtype=np.int32)
    actions = np.zeros([100, num_component], dtype=np.int32)
    years = np.zeros([100, 22], dtype=np.int32)
    rewards = np.zeros([100, num_component], dtype=np.float32)

    while not done:
        rnd = np.random.rand()
        states[t, :] = copy.deepcopy(s[0])
        a = np.zeros(num_component, dtype=np.int32)
        if rnd < e and i_episode%10!=0:
            ind = np.random.choice(range(num_component), np.random.randint(0, num_component))
            a[ind] = np.random.randint(0, 4, ind.shape[0])
        else:
            a = sess.run(agent.actions, feed_dict={agent.s: s, agent.bin_year: year})[0]
        state1, reward, done = env.step(a)
        r = reward/600

        s1 = env.state[np.newaxis,:num_component]
        states_[t] = copy.deepcopy(s1)
        rewards[t] = copy.deepcopy(r)
        actions[t] = copy.deepcopy(a)
        years[t] = copy.deepcopy(year[0])
        s = s1
        t += 1
        year = time_encoder(t, 22)[np.newaxis, :]
        rAll += np.sum(r)
    print(rAll)
    G = getreturn(rewards, gamma)
    costs.append(rAll)

    for i in range(100):
        s = states[i]; a = actions[i]; Q = G[i]; s1 = states_[i]; r = rewards[i]; year = years[i]
        for component in range(num_component):
            tuple_idx = (component, s[component], a[component], i)
            if Q_dict.get(tuple_idx):
                n_times,Q_value = Q_dict[tuple_idx]
                n_times += 1
                Q_value += (Q[component]-Q_value)/n_times
                Q_dict[tuple_idx] = (n_times, Q_value)
            else:
                n_times = 0
                Q_value = Q[component]
                Q_dict[tuple_idx] = (n_times, Q_value)
            Q[component] = Q_value
        Buffer.add(np.reshape(np.array([s, a, s1, Q, r, year]), [1, 6]))

    if i_episode > 100 and i_episode%update_freq==0:
        if e > endE: e -= e_step
        for m in range(1):
            trainBatch = Buffer.sample(batch_size)
            train_s = np.vstack(trainBatch[:, 0])
            train_a = np.vstack(trainBatch[:, 1])
            train_year = np.vstack(trainBatch[:, -1])
            # train_year = train_year[:, :]
            target_Q = np.vstack(trainBatch[:, 3])*100
            _, qloss = sess.run([agent.train, agent.loss], feed_dict={agent.s: train_s, agent.bin_year:train_year,
                                                                      agent.pre_act: train_a, agent.target_Q:target_Q})
        DQNloss.append(qloss)
    print(i_episode, time.time() - start, e, qloss, rAll)

    if i_episode%5000==0:
        filepath = './result/training results/step' + str(i_episode)
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        saver.save(sess, filepath + '/training-' + str(i_episode) + '.cpkt')
        np.save(filepath + '/costs.npy', costs)
        np.save(filepath + '/DQNloss.npy', DQNloss)
        # np.save(filepath + '/Q_dict.npy', Q_dict)
        print("Save Model")
        elapsed = time.time() - start
        print(i_episode, e, elapsed, costs[-1])
        state, _ = env.randomint()
        print(state[0:10])
        s = state[np.newaxis, :num_component]
        year = time_encoder(state[-1], 22)[np.newaxis, :]
        print(sess.run(agent.actions, feed_dict={agent.s: s, agent.bin_year:year})[0,0:10])
        start = time.time()
    # if i_episode%5000==0:
        with open('./result/Qvalue.pickle','wb') as saved:
            pickle.dump(Q_dict, saved)
            # np.save('./result/Buffer.npy', Buffer)


