# policy network
import tensorflow as tf
import tensorflow.contrib.slim as slim

class CNNagent(object):
    def __init__(self, hidden_size=8000, num_components=263, lr=0.001):
        self.s = tf.placeholder(shape=[None, num_components], dtype=tf.int32)
        self.bin_year = tf.placeholder(shape=[None, 22], dtype=tf.float32)
        self.s_onehot = tf.one_hot(self.s, 6, dtype=tf.float32)
        self.s_onehot = tf.reshape(self.s_onehot, [-1, num_components*6])
        self.s_ = tf.concat([self.s_onehot, self.bin_year], axis=1)
        self.input_s = tf.reshape(self.s_, [-1, 40, 40, 1])

        self.conv1 = slim.conv2d(inputs=self.input_s, num_outputs=8, kernel_size=[3, 3], stride=[1, 1], padding='SAME',
                                 activation_fn=tf.nn.relu, biases_initializer=tf.random_normal_initializer(0.))
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=32, kernel_size=[3, 3], stride=[2, 2], padding='SAME',
                                 activation_fn=tf.nn.relu, biases_initializer=tf.random_normal_initializer(0.))
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=128, kernel_size=[3, 3], stride=[2, 2], padding='SAME',
                                 activation_fn=tf.nn.relu, biases_initializer=tf.random_normal_initializer(0.))
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=512, kernel_size=[3, 3], stride=[2, 2], padding='SAME',
                                 activation_fn=tf.nn.relu, biases_initializer=tf.random_normal_initializer(0.))
        self.conv5 = slim.conv2d(inputs=self.conv4, num_outputs=1024, kernel_size=[3, 3], stride=[2, 2], padding='SAME',
                                 activation_fn=tf.nn.relu, biases_initializer=tf.random_normal_initializer(0.))
        self.conv6 = slim.conv2d(inputs=self.conv5, num_outputs=4096, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
                                 activation_fn=tf.nn.relu, biases_initializer=tf.random_normal_initializer(0.))


        self.layer1 = slim.flatten(self.conv6)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.layer2 = tf.layers.dense(inputs=self.layer1, units=hidden_size, activation=tf.nn.relu)
        self.streamA, self.streamC = tf.split(value=self.layer2, num_or_size_splits=2, axis=1)
        self.Advantage = tf.layers.dense(inputs=self.streamA, units=num_components*4, activation=None,
                                         kernel_initializer=xavier_init, bias_initializer=None)
        self.Advantage = tf.reshape(self.Advantage, [-1, num_components, 4])
        self.Value = tf.layers.dense(inputs=self.streamC, units=num_components, activation=None,
                                     kernel_initializer=xavier_init, bias_initializer=None)
        self.Value = tf.reshape(self.Value, [-1, num_components, 1])
        self.Q_out = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=2, keepdims=True))
        self.actions = tf.cast(tf.argmax(self.Q_out, axis=2), dtype=tf.int32)

        self.target_Q = tf.placeholder(shape=[None, num_components], dtype=tf.float32)
        self.pre_act = tf.placeholder(shape=[None, num_components], dtype=tf.int32)
        self.pre_act_onehot = tf.one_hot(indices=self.pre_act, depth=4, dtype=tf.float32)

        self.Q = tf.reduce_mean(tf.multiply(self.Q_out, self.pre_act_onehot), axis=2)
        self.td_error = tf.reduce_mean(tf.square(self.target_Q - self.Q))
        self.loss = tf.reduce_mean(self.td_error)
        self.train = tf.train.AdamOptimizer(learning_rate=lr).minimize(self.loss)

