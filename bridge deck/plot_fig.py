# **************************************************************************************************************************
"costs & loss figures"
import numpy as np
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
import matplotlib.pyplot as plt
font = {"fontname": 'Times New Roman', "fontsize": 16}
font1 = {"fontname": 'Times New Roman', "fontsize": 12}
costs = np.load('./result/training results/step90000/costs.npy')
loss = np.load('./result/training results/step90000/DQNloss.npy')
A = np.reshape(costs[1:], [900,100])
B = np.reshape(loss, [899,100])
fig, ax1 = plt.subplots()
ax1.plot(np.arange(1,901), -np.log10(-np.mean(A, 1)),linewidth=2, color='g')
plt.yticks([-np.log10(30), -np.log10(20), -np.log10(10), -np.log10(5), -np.log10(4), -np.log10(3), -np.log10(2),-np.log10(1)],
           ['-30','-20','-10','-5','-4','-3','-2','-1'])
ax2 = ax1.twinx()
ax2.plot(np.arange(1,900), np.log10(np.mean(B, 1)),linewidth=2, color='b')
plt.yticks([np.log10(0.0001), np.log10(0.001), np.log10(0.01), np.log10(0.1), np.log10(1)],
           ['0.0001','0.001','0.01','0.1','1'])
ax1.set_xlabel('Training steps', **font)
ax1.set_ylabel('Total costs', **font, color = 'g')
ax2.set_ylabel('DQN loss', **font, color = 'b')
plt.xlim([-50, 950])
plt.xticks(np.arange(0,901, 300), ['0', '30000', '60000', '90000'], **font1)
plt.savefig('./figures/cost&loss.pdf')
plt.savefig('./figures/cost&loss.eps')


# *********************************************************************************************************************
static = True
while static:
    font = {"fontname": 'Times New Roman', "fontsize": 16}
    font1 = {'fontname':'times new roman', 'fontsize':12}
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('font', family='Times New Roman')
    # plot test distribution of diferent policies
    states = np.load('./result/simulation/DQN/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    name = 'DRL'
    plt.figure(figsize=(20, 5));    plt.subplot(241);    plt.bar(x, z/7e5, label='DRL'); plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    states = np.load('./result/simulation/c-1/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    name = 'Condition-1'
    plt.subplot(242);    plt.bar(x, z/7e5, label='Condition-1'); plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font); plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    states = np.load('./result/simulation/c-2/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(243);    plt.bar(x, z/7e5, label='Condition-2');  plt.legend(loc='upper right');    plt.ylim([0, 600])
    plt.xlabel('Conditions', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    states = np.load('./result/simulation/c-3/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(244);    plt.bar(x, z/7e5, label='Condition-3');  plt.legend(loc='upper right');    plt.ylim([0, 600])
    plt.xlabel('Conditions', **font);    plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    states = np.load('./result/simulation/t-5/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(245);    plt.bar(x, z/7e5, label='Time-5');    plt.legend(loc='upper right');  plt.ylim([0, 600])
    plt.xlabel('Conditions', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    states = np.load('./result/simulation/t-10/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(246);    plt.bar(x, z/7e5, label='Time-10');  plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font); plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    states = np.load('./result/simulation/t-15/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(247);    plt.bar(x, z/7e5, label='Time-15');    plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font);plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    states = np.load('./result/simulation/t-20/states-999.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(248);    plt.bar(x, z/7e5, label='Time-20');    plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font);plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/7e5)
        plt.annotate(aa, (idx - dd, z[idx]/7e5 + 0.01), **font1)

    static = False
plt.savefig('./figures/statistic.eps')
plt.savefig('./figures/statistic.pdf')

## ********************************************************************************************************************
# ********************************************************************************************************
# action static
statistic = True
while statistic:
    font = {"fontname": 'Times New Roman', "fontsize": 16}
    font1 = {'fontname':'times new roman', 'fontsize':12}
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('font', family='Times New Roman')
    aa = []
    for i in np.arange(0, 101, 20):
        aa.append(str(i))
    # plot test distribution of diferent policies
    states = np.load('./result/simulation/DQN/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    name = 'DRL'
    plt.figure(figsize=(18, 5));    plt.subplot(241);    plt.bar(x, z/np.sum(z), label='DRL:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .1])
    plt.xticks(np.arange(0,21,4), aa)


    states = np.load('./result/simulation/c-1/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(242);    plt.bar(x, z/np.sum(z), label='Condition-1:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .1])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulationl/c-2/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(243);    plt.bar(x, z/np.sum(z), label='Condition-2:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .1])
    plt.xticks(np.arange(0,21,4)-0.5, aa)


    states = np.load('./result/simulation/c-3/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(244);    plt.bar(x, z/np.sum(z), label='Condition-3:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .1])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulation/t-5/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(245);    plt.bar(x, z/np.sum(z), label='Time-5:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Seving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .08])
    plt.xticks(np.arange(0,21,4)-0.5, aa)


    states = np.load('./result/simulation/t-10/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(246);    plt.bar(x, z/np.sum(z), label='Time-10:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .15])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulation/t-15/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(247);    plt.bar(x, z/np.sum(z), label='Time-15:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .22])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulation/t-20/actions-999.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(248);    plt.bar(x, z/np.sum(z), label='Time-20:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .3])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    statistic = False
plt.savefig('./figures/statistic_action.eps')
plt.savefig('./figures/statistic_action.pdf')

# ********************************************************************************************************************
action_show = True
while action_show:
    font = {"fontname": 'Times New Roman', "fontsize": 16}
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rc('font', family='Times New Roman')
    # plot test distribution of diferent policies
    actions = np.load('./result/DQN simulation/training-step89300/actions-999.npy')
    plt.figure(figsize=(10,9));plt.subplot(311); plt.title('DRL',**font)
    for i in range(7):
        plt.plot(actions[9, :, i], '.-')
    plt.ylabel('Maintenance action',**font); plt.grid()


    actions = np.load(
        './result/DQN simulation/manual/sim-c1-99/actions-99.npy')
    plt.subplot(312); plt.title('Condition-1',**font)
    for i in range(7):
        plt.plot(actions[9, :, i], '.-')
    plt.ylabel('Maintenance action',**font); plt.grid()


    actions = np.load(
        './result/DQN simulation/manual/sim15-99/actions-99.npy')
    plt.subplot(313); plt.title('Time-15-year',**font)
    for i in range(7):
        plt.plot(actions[9, :, i], '.-')
    plt.ylabel('Maintenance action',**font); plt.xlabel('Time steps (year)', **font); plt.grid()
    action_show = False


# *******************************************************************************************************************
import numpy as np
import tensorflow as tf
from policynet import CNNPolicy
import matplotlib.pyplot as plt
import scipy.io as sio
import time, os
from bridgedeterioration import BridgeEnv
from utils import  plot_state
env = BridgeEnv()
start = time.time()

font = {"fontname": 'Times New Roman', "fontsize": 16}
path = "./result/training results"  # The path to save our model to.
simulation_path = "./result/simulation"

tf.reset_default_graph()
mH = 128
mainQN = CNNPolicy(); mainQN.create_network(mH)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

base = [89300] #,3981,15848]
# simulation from trained models
for step in base:
    result_path = path + '/step' + str(step)
    sess = tf.Session()
    sess.run(init)
    ckpt = tf.train.get_checkpoint_state(result_path)
    saver.restore(sess, ckpt.model_checkpoint_path)  # load trained model

    costs = []
    num_episodes = 1
    actions = np.zeros([num_episodes,100,7], dtype=np.int32)
    states = np.zeros([num_episodes,101,7], dtype=np.int32)

    path1 = simulation_path + '/training-step' + str(step)
    if os.path.exists(path1)==False:
        os.mkdir(path1)
    for i_episode in range(num_episodes):
        t=0
        rAll = 0
        done  = False
        current_state = env.reset()
        a = []
        states[i_episode,0] = env.state_num
        while not done:
            plt.figure(1)
            plot_state(current_state, 7, 7)
            plt.xlabel('states', **font)
            plt.ylabel('components', **font)
            state = np.reshape(current_state,[49])
            action = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [state]})[0]
            next_state, next_reward, done = env.step(action)
            next_reward = next_reward/600
            rAll += np.sum(next_reward)
            actions[i_episode,t] = action
            states[i_episode,t+1] = env.state_num

            plt.title('action:' + str(action)  + '  time:' + str(env.time) + '\n' + 'reward:' +
                      np.array2string(next_reward,precision=2) +' total costs:' + str("%.2f"%rAll),**font)
            plt.savefig(path1 + '/' + str(t) + '.eps')
            # # plt.savefig(c_path + '/' + str(t) + '.tif')
            # plt.savefig(path1 + '/' + str(t) + '.png')
            plt.close()
            t += 1
            current_state = next_state
        costs.append(rAll)
np.save(path1 + '/actions-'+str(i_episode)+'.npy', actions)
np.save(path1 + '/states-' + str(i_episode) + '.npy', states)
np.save(path1 + '/costs-'+str(i_episode)+'.npy', costs)

sess.close()

# manual maintenance solution
# minor maintenance every 5 years


path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1000
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim5'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):
    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        # plt.figure(1)
        # plot_state(current_state, 7, 7)
        # plt.xlabel('states', **font)
        # plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        if t>0 and (t%5 == 0):
            action = np.ones(7,dtype=np.int32)
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        # plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
        #           np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        # plt.savefig(c_path + '/' + str(t) + '.eps')
        # plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)


# manual maintenance solution
# minor maintenance every 10 years

path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1000
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim10'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):
    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        # plt.figure(1)
        # plot_state(current_state, 7, 7)
        # plt.xlabel('states', **font)
        # plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        if t>0 and (t%10 == 0):
            action = np.ones(7,dtype=np.int32)
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        # plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
        #           np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        # plt.savefig(c_path + '/' + str(t) + '.eps')
        # # plt.savefig(c_path + '/' + str(t) + '.tif')
        # plt.savefig(c_path + '/' + str(t) + '.png')
        #
        # plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)


# manual maintenance solution
# minor maintenance every 15 years

path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim15'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):

    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        plt.figure(1)
        plot_state(current_state, 7, 7)
        plt.xlabel('states', **font)
        plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        if t>0 and (t%15 == 0):
            action = np.ones(7,dtype=np.int32)
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
                  np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        plt.savefig(c_path + '/' + str(t) + '.eps')
        # # plt.savefig(c_path + '/' + str(t) + '.tif')
        # plt.savefig(c_path + '/' + str(t) + '.png')
        plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)


# manual maintenance solution
# minor maintenance every 20 years

path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1000
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim20'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):
    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        # plt.figure(1)
        # plot_state(current_state, 7, 7)
        # plt.xlabel('states', **font)
        # plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        if t>0 and (t%20 == 0):
            action = np.ones(7,dtype=np.int32)
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        # plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
        #           np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        # plt.savefig(c_path + '/' + str(t) + '.eps')
        # # plt.savefig(c_path + '/' + str(t) + '.tif')
        # plt.savefig(c_path + '/' + str(t) + '.png')
        # plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)


# manual maintenance solution
# mcondition-based solution;  minor repair when condition worse than 4

path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1000
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim-c4'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):
    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        # plt.figure(1)
        # plot_state(current_state, 7, 7)
        # plt.xlabel('states', **font)
        # plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        action[np.where(env.state_num>4)] = 1
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        # plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
        #           np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        # plt.savefig(c_path + '/' + str(t) + '.eps')
        # # plt.savefig(c_path + '/' + str(t) + '.tif')
        # plt.savefig(c_path + '/' + str(t) + '.png')
        # plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)



# manual maintenance solution
# mcondition-based solution;  minor repair when condition worse than 3

path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1000
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim-c3'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):
    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        # plt.figure(1)
        # plot_state(current_state, 7, 7)
        # plt.xlabel('states', **font)
        # plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        action[np.where(env.state_num>3)] = 1
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        # plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
        #           np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        # plt.savefig(c_path + '/' + str(t) + '.eps')
        # # plt.savefig(c_path + '/' + str(t) + '.tif')
        # plt.savefig(c_path + '/' + str(t) + '.png')
        # plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)



# manual maintenance solution
# mcondition-based solution;  minor repair when condition worse than 2

path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1000
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim-c2'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):
    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        # plt.figure(1)
        # plot_state(current_state, 7, 7)
        # plt.xlabel('states', **font)
        # plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        action[np.where(env.state_num>2)] = 1
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        # plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
        #           np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        # plt.savefig(c_path + '/' + str(t) + '.eps')
        # # plt.savefig(c_path + '/' + str(t) + '.tif')
        # plt.savefig(c_path + '/' + str(t) + '.png')
        # plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)


# manual maintenance solution
# mcondition-based solution;  minor repair when condition worse than 1

path1 = './result/DQN simulation/manual'
costs = []
num_episodes = 1
actions = np.zeros([num_episodes, 100, 7], dtype=np.int32)
states = np.zeros([num_episodes, 101, 7], dtype=np.int32)
c_path = path1 + '/sim-c1'
if os.path.exists(c_path) == False:
    os.mkdir(c_path)
for i_episode in range(num_episodes):
    t = 0
    current_state = env.reset()
    states[i_episode,0] = env.state_num
    rAll = 0
    done = False

    while not done:
        plt.figure(1)
        plot_state(current_state, 7, 7)
        plt.xlabel('states', **font)
        plt.ylabel('components', **font)
        state = np.reshape(current_state, [49])
        action = np.zeros(7, dtype=np.int32)
        action[np.where(env.state_num>1)] = 1
        next_state, next_reward, done = env.step(action)
        next_reward = next_reward / 600
        rAll += sum(next_reward)
        actions[i_episode,t] = action
        states[i_episode,t+1] = env.state_num
        plt.title('action:' + str(action) + '  time:' + str(env.time) + '\n' + 'reward:' +
                  np.array2string(next_reward, precision=2) + ' total costs:' + str("%.2f" % rAll), **font)
        plt.savefig(c_path + '/' + str(t) + '.eps')
        # # plt.savefig(c_path + '/' + str(t) + '.tif')
        # plt.savefig(c_path + '/' + str(t) + '.png')
        plt.close()
        current_state = next_state
        t += 1
    costs.append(rAll)
np.save(c_path + '/actions-' + str(i_episode) + '.npy', actions)
np.save(c_path + '/states-' + str(i_episode) + '.npy', states)
np.save(c_path + '/costs-' + str(i_episode) + '.npy', costs)
np.mean(costs)
