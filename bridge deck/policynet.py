from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim


class CNNPolicy(object):
    def create_network(self,mH=128):
        self.scalarInput = tf.placeholder(shape=[None,49], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1,7,7,1])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=4, kernel_size=[3,3], stride=[1,1],
                                 activation_fn=tf.nn.tanh, padding='SAME', biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=16, kernel_size= [3,3], stride=[2,2],
                                 activation_fn=tf.nn.tanh,padding='SAME', biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=32, kernel_size= [3,3], stride=[2,2],
                                 activation_fn=tf.nn.tanh,padding='SAME', biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=mH, kernel_size= [2,2], stride=[1,1],
                                 activation_fn=tf.nn.tanh,padding='VALID', biases_initializer=None)

        # duel DQN, outputs concludes advantage and value streams
        self.layer4 = slim.flatten(self.conv4)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.W1 = tf.Variable(xavier_init([mH,mH]))
        self.b1 = tf.Variable(tf.zeros([mH]))
        self.layer5 = tf.nn.relu(tf.matmul(self.layer4,self.W1)+self.b1)
        self.W2 = tf.Variable(xavier_init([mH,mH]))
        self.b2 = tf.Variable(tf.zeros([mH]))
        self.layer6 = tf.nn.relu(tf.matmul(self.layer5,self.W2)+self.b2)
        self.streamA, self.streamV = tf.split(self.layer6,2,1)  # tf.split(data, number, axis)

        # 4 actions
        self.AW = tf.Variable(xavier_init([mH//2, 28]))  # AW [h_size//2, 7*4]
        self.Ab = tf.Variable(tf.zeros([28]))
        self.VW = tf.Variable(xavier_init([mH//2, 7]))  # value V(s)
        self.Vb = tf.Variable(tf.zeros([7]))
        self.Advantage = tf.matmul(self.streamA, self.AW)+self.Ab
        self.Value = tf.matmul(self.streamV, self.VW)+self.Vb
        self.Advantage = tf.reshape(self.Advantage, [-1, 7, 4])
        self.Value = tf.reshape(self.Value, [-1,7,1])

        # combine advantage and value network together
        self.Qout = self.Value+tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=2, keep_dims=True))  # 1*7*4 --> Q(s,a)
        self.predict = tf.argmax(self.Qout, 2)  # the predicted actions for each component 1*7*1 --> actions
        # self.predict_a = tf.nn.softmax(self.Qout,2)

        self.targetQ = tf.placeholder(shape=[None,7], dtype = tf.float32)
        self.actions = tf.placeholder(shape=[None,7], dtype = tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, 4, dtype = tf.float32)

        self.Q = tf.reduce_mean(tf.multiply(self.Qout, self.actions_onehot),axis=2)
        self.td_error = tf.reduce_mean(tf.square(self.targetQ-self.Q))
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate = 0.001)  # training rules
        self.updateModel = self.trainer.minimize(self.loss)  # training target
