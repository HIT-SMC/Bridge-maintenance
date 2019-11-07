import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
import os, time, copy
from bridgedeterioration import BridgeEnv
from policynet import CNNPolicy
from utils import getreturn, experience_buffer, processsa
# from pretrain import  pretrain
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
env = BridgeEnv()
start = time.time()

batch_size = 1000; update_freq = 1
startE = 1.0; endE = 0.01; anneling_steps = int(1e2); e_step = (startE-endE)/anneling_steps; e = startE
num_episodes = int(1e4); pre_train_steps = int(1e4)
loadmodel = False

tf.reset_default_graph()
mH = 128; gamma = 1.0; alpha = 1.0
mainQN = CNNPolicy(); mainQN.create_network(mH)
init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=1000)
myBuffer = experience_buffer(pre_train_steps)

costs = [0]; DQNloss = [0]; Q_dict = {}
if not os.path.exists('./result'):
    os.mkdir('./result')
networkmodel = './result/training result/'

# guidebuffer = pretrain(gamma)
sess = tf.Session()
sess.run(init)

if loadmodel:
    ckpt = tf.train.get_checkpoint_state(networkmodel)
    saver.restore(sess, ckpt.model_checkpoint_path)

for i_episode in range(num_episodes+1):
    episodeBuffer = experience_buffer(pre_train_steps)

    state = env.reset()
    s = np.reshape(state,[49])
    done = False
    rAll = 0; t = 0
    states = np.zeros([100,49])
    actions = np.zeros([100,7])
    codeactions = np.zeros([100,28])
    states1 = np.zeros([100,49])
    rewards = np.zeros([100,7])
    dones = np.zeros([100,1])
    rnd = np.random.rand()

    while not done:
        states[t] = copy.deepcopy(s)

        a = np.zeros(7,dtype=np.int32)
        if rnd < e:
            a = np.random.randint(0,4,7)
        else:
            a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput:[s]})[0]

        state1, reward, done = env.step(a)
        r = reward/600
        s1 = np.reshape(state1,[49])
        code_a = np.reshape(processsa(a,4),[28])

        states1[t] = copy.deepcopy(s1); rewards[t] = copy.deepcopy(r); actions[t] = copy.deepcopy(a)
        codeactions[t] = copy.deepcopy(code_a); dones[t] = copy.deepcopy(done)
        s = s1; t+=1
    G = getreturn(rewards,gamma=gamma)
    costs.append(np.sum(rewards))
    if i_episode%100==0:
        print(str(i_episode), '***********************', costs[-1],  DQNloss[-1], '***********************', time.time() - start)

    for i in range(100):
        s = states[i]; a = actions[i]; Q = G[i]; s1 = states1[i]; d = dones[i]; code_a = codeactions[i]; r = rewards[i]
        state = np.int32(np.reshape(s,[7,7]))
        _, state_num = np.where(state[:,0:6]==1)
        a = np.int32(a)
        for component in range(7):
            tuple_idx = (component, state_num[component], a[component], i)
            if Q_dict.get(tuple_idx):
                n_times,Q_value = Q_dict[tuple_idx]
                n_times += 1
                Q_value += (Q[component]-Q_value)*alpha
                Q_dict[tuple_idx] = (n_times, Q_value)
            else:
                n_times = 0
                Q_value = Q[component]
                Q_dict[tuple_idx] = (n_times, Q_value)
            Q[component] = Q_value
        episodeBuffer.add(np.reshape(np.array([s,a,Q,r,s1,d,code_a]),[1,7]))
    myBuffer.add(episodeBuffer.buffer)

    if i_episode >100 and (i_episode%update_freq == 0):
        if e>endE:
            e -= e_step
        trainBatch = myBuffer.sample(batch_size)
        input_s = np.vstack(trainBatch[:,0])
        input_a = np.vstack(trainBatch[:,1])
        target_Q = np.vstack(trainBatch[:,2])

        _, qloss = sess.run([mainQN.updateModel,mainQN.loss],
                 feed_dict={mainQN.scalarInput:input_s, mainQN.actions:input_a, mainQN.targetQ:target_Q})
        DQNloss.append(qloss)

    if i_episode%5000 == 0:
        print(rnd>e)
        filepath = './result/training results/step' + str(i_episode)
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        saver.save(sess, filepath+'/training-'+str(i_episode)+'.cpkt')
        np.save(filepath+'/costs.npy', costs)
        np.save(filepath+'/DQNloss.npy', DQNloss)
        np.save(filepath+'/Q_dict.npt',Q_dict)
        print("Save Model")
        elapsed = time.time()-start
        print(i_episode,e,elapsed,costs[-1])
        state,_ = env.randomint()
        print(state)
        s = np.reshape(state,[49])
        print(sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0])
        start = time.time()
sess.close()

