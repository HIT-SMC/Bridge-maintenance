# **************************************************************************************************************************
"costs & loss figures"
import numpy as np
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
import matplotlib.pyplot as plt
font = {"fontname": 'Times New Roman', "fontsize": 16}
font1 = {"fontname": 'Times New Roman', "fontsize": 12}
costs = np.load('./result/training results/step110000/costs.npy')
loss = np.load('./result/training results/step110000/DQNloss.npy')
A = np.reshape(costs[1:], [1099,100])
B = np.reshape(loss, [1099,100])
fig, ax1 = plt.subplots()
ax1.plot(np.arange(1,1100), -np.log10(-np.max(A, 1)-0.2),linewidth=2, color='g')
plt.yticks([ -np.log10(10), -np.log10(5), -np.log10(4), -np.log10(3), -np.log10(2), -np.log10(1)],
           ['-10','-5','-4','-3','-2','-1'])
ax2 = ax1.twinx()
ax2.plot(np.arange(1,1100), np.log10(np.mean(B, 1)),linewidth=2, color='b')
plt.yticks([np.log10(0.00001), np.log10(0.0001), np.log10(0.001), np.log10(0.01), np.log10(0.1)],
           ['0.00001','0.0001','0.001','0.01','0.1'])
ax1.set_xlabel('Training steps', **font)
ax1.set_ylabel('Total costs', **font, color = 'g')
ax2.set_ylabel('DQN loss', **font, color = 'b')
plt.xlim([-50, 1150])
plt.xticks(np.arange(0,1150, 200), ['0', '20000', '40000', '60000','80000','100000'], **font1)
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
    states = np.load('./result/simulation/DQN/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    name = 'DRL'
    plt.figure(figsize=(20, 5));    plt.subplot(241);    plt.bar(x, z/2.63e7, label='DRL'); plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

    states = np.load('./result/simulation/c-1/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    name = 'Condition-1'
    plt.subplot(242);    plt.bar(x, z/2.63e7, label='Condition-1'); plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font); plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

    states = np.load('./result/simulation/c-2/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(243);    plt.bar(x, z/2.63e7, label='Condition-2');  plt.legend(loc='upper right');    plt.ylim([0, 600])
    plt.xlabel('Conditions', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

    states = np.load('./result/simulation/c-3/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(244);    plt.bar(x, z/2.63e7, label='Condition-3');  plt.legend(loc='upper right');    plt.ylim([0, 600])
    plt.xlabel('Conditions', **font);    plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

    states = np.load('./result/simulation/t-5/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(245);    plt.bar(x, z/2.63e7, label='Time-5');    plt.legend(loc='upper right');  plt.ylim([0, 600])
    plt.xlabel('Conditions', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

    states = np.load('./result/simulation/t-10/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(246);    plt.bar(x, z/2.63e7, label='Time-10');  plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font); plt.ylabel('Frequency', **font);   plt.ylim([0, .8])
    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

    states = np.load('./result/simulation/t-15/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(247);    plt.bar(x, z/2.63e7, label='Time-15');    plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font);plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

    states = np.load('./result/simulation/t-20/states.npy')
    x = range(0, 6);    z = np.zeros(6, dtype=np.int32)
    for idx in x:
        z[idx] = len(np.where(states[:, 0:100, :] == idx)[0])
    plt.subplot(248);    plt.bar(x, z/2.63e7, label='Time-20');    plt.legend(loc='upper right')
    plt.xlabel('Conditions', **font);plt.ylabel('Frequency', **font);   plt.ylim([0, .8])

    for idx in x:
        if z[idx] > 99:
            dd = 0.35
        elif z[idx] > 9:
            dd = 0.2
        else:
            dd = 0.1
        aa = '%0.3f' % (z[idx]/2.63e7)
        plt.annotate(aa, (idx - dd, z[idx]/2.63e7 + 0.01), **font1)

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
    states = np.load('./result/simulation/DQN/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    name = 'DRL'
    plt.figure(figsize=(18, 5));    plt.subplot(241);    plt.bar(x, z/np.sum(z), label='DRL:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .08])
    plt.xticks(np.arange(0,21,4), aa)


    states = np.load('./result/simulation/c-1/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(242);    plt.bar(x, z/np.sum(z), label='Condition-1:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .1])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulation/c-2/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(243);    plt.bar(x, z/np.sum(z), label='Condition-2:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .1])
    plt.xticks(np.arange(0,21,4)-0.5, aa)


    states = np.load('./result/simulation/c-3/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(244);    plt.bar(x, z/np.sum(z), label='Condition-3:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .1])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulation/t-5/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(245);    plt.bar(x, z/np.sum(z), label='Time-5:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Seving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .08])
    plt.xticks(np.arange(0,21,4)-0.5, aa)


    states = np.load('./result/simulation/t-10/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(246);    plt.bar(x, z/np.sum(z), label='Time-10:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .15])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulation/t-15/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(247);    plt.bar(x, z/np.sum(z), label='Time-15:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .22])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    states = np.load('./result/simulation/t-20/actions.npy')
    x = np.arange(0, 20);    z = np.zeros(20, dtype=np.int32)
    for idx in range(20):
        z[idx] = len(np.where(states[:, idx*5:(idx+1)*5, :] != 0)[0])
    plt.subplot(248);    plt.bar(x, z/np.sum(z), label='Time-20:%d' % np.sum(z)); plt.legend(loc='upper right')
    plt.xlabel('Serving year', **font);     plt.ylabel('Frequency', **font);   plt.ylim([0, .3])
    plt.xticks(np.arange(0,21,4)-0.5, aa)

    statistic = False
plt.savefig('./figures/statistic_action.eps')
plt.savefig('./figures/statistic_action.pdf')

