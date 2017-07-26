import tensorflow as tf
import pickle
import numpy as np

class model:
    def __init__(self, path):
        self.data_dict = {}
        with open(path, 'rb') as f:
            self.data_dict = pickle.load(f)
        print('model weight file loaded')

    def build(self):
        self.images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])

        conv1 = self.conv_layer(self.images, 64, 'conv1zzz')
        pool1 = self.max_pool(conv1)

        conv2 = self.conv_layer(pool1, 128, 'conv2d')
        pool2 = self.max_pool(conv2)

        conv3_1 = self.conv_layer(pool2, 256, 'conv2d_1')
        conv3_2 = self.conv_layer(conv3_1, 256, 'conv2d_2')
        pool3 = self.max_pool(conv3_2)

        conv4_1 = self.conv_layer(pool3, 512, 'conv2d_3')
        conv4_2 = self.conv_layer(conv4_1, 512, 'conv2d_4')
        pool4 = self.max_pool(conv4_2)

        fc = self.fc_layer(pool4, 4096, 'dense',  if_reshape=True)
        y_result = self.fc_layer(fc, 3755, 'dense_1')

        prob = tf.nn.softmax(y_result)
        self.val_top_k, self.index_top_k = tf.nn.top_k(prob, k=5)

        self.data_dict = None
        print('build model finish!')

    def conv_layer(self, input, filters, name, ksize=3):
        with tf.variable_scope(name):
            str1 = name + '/kernel:0'
            str2 = name + '/bias:0'
            return tf.layers.conv2d(inputs=input, filters=filters, kernel_size=[ksize, ksize], padding='same',
                                    activation=tf.nn.relu, kernel_initializer=self.get_weight(str1),
                                    bias_initializer=self.get_weight(str2),
                                    trainable=False)

    def max_pool(self, input, name=None, stride=2):
            return tf.layers.max_pooling2d(inputs=input, pool_size=[2, 2], strides=stride, padding='same', name=name)

    def fc_layer(self, input, unit, name, if_reshape=False):
        with tf.variable_scope(name):
            if if_reshape:
                shape = input.get_shape().as_list()
                dim = 1
                for d in shape[1:]:
                    dim *= d
                input = tf.reshape(input, [-1, dim])
            str1 = name + '/kernel:0'
            str2 = name + '/bias:0'    
            fc = tf.layers.dense(inputs=input, units=unit, activation=tf.nn.relu,
                                   kernel_initializer=self.get_weight(str1),
                                   bias_initializer=self.get_weight(str2))
            return fc

    def get_weight(self, name):
        return tf.constant_initializer(self.data_dict[name])        