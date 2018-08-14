import argparse
import sys
import pickle
import numpy as np

import cifar10_utils

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

class VGG:
    def __init__(self, dataset):
        if data == 'cifar10':
            self.num_classes = 10

        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        self.label = tf.placeholder(tf.int32, [None, self.num_classes], name='label')

        self.logits = self.load_model()
        self.model = tf.identity(self.logits, name='logits')

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        self.correct_pred = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32), name='accuracy')

    """
        types
        A : 11 weight layers
        A-LRN : 11 weight layers with Local Response Normalization
        B : 13 weight layers
        C : 16 weight layers with 1D conv layers 
        D : 16 weight layers
        E : 19 weight layers
    """
    def load_model(self, model_type='A'):
        # LAYER GROUP #1
        group_1 = conv2d(self.input, num_outputs=64,
                    kernel_size=[3,3], stride=1, padding='SAME',
                    activation_fn=tf.nn.relu)
        
        if model_type == 'A-LRN':
            group_1 = tf.nn.local_response_normalization(conv1, bias=2, alpha=0.0001,beta=0.75)

        if model_type != 'A' && model_type == 'A-LRN':
            group_1 = conv2d(group_1, num_outputs=64,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_1 = max_pool2d(group_1, kernel_size=[2,2], stride=2)

        # LAYER GROUP #2
        group_2 = conv2d(group_1, num_outputs=128,
                            kernel_size=[3, 3], padding='SAME',
                            activation_fn=tf.nn.relu)

        if model_type != 'A' && model_type == 'A-LRN':
            group_2 = conv2d(group_2, num_outputs=128,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)   

        group_2 = max_pool2d(group_2, kernel_size=[2,2], stride=2)

        # LAYER GROUP #3
        group_3 = conv2d(group_2, num_outputs=256,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)    
        group_3 = conv2d(group_3, num_outputs=256,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)

        if model_type == 'C':
            # group_3 = tf.layers.conv1d(group_3, filters=256, )

        if model_type == 'D' || model_type == 'E':
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)     

        if model_type == 'E':
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_3 = max_pool2d(group_3, kernel_size=[2,2], stride=2)

        # LAYER GROUP #4
        group_4 = conv2d(group_3, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)
        group_4 = conv2d(group_4, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)    

        if model_type == 'C':
            # group_4 = tf.layers.conv1d(group_4, filters=256, )

        if model_type == 'D' || model_type == 'E':
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)     

        if model_type == 'E':
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_4 = max_pool2d(group_4, kernel_size=[2,2], stride=2)

        # LAYER GROUP #5
        group_5 = conv2d(group_4, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)
        group_5 = conv2d(group_5, num_outputs=512,
                            kernel_size=[3,3], stride=1, padding='SAME',
                            activation_fn=tf.nn.relu)    

        if model_type == 'C':
            # group_5 = tf.layers.conv1d(group_5, filters=256, )

        if model_type == 'D' || model_type == 'E':
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)     

        if model_type == 'E':
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_5 = max_pool2d(group_5, kernel_size=[2,2], stride=2)

        # 1st FC 4096
        flat = flatten(group_5)
        fcl1 = fully_connected(flat, num_outputs=4096, activation_fn=tf.nn.relu)
        dr1 = tf.nn.dropout(fcl1, 0.5)

        # 2nd FC 4096
        fcl2 = fully_connected(dr1, num_outputs=4096, activation_fn=tf.nn.relu)
        dr2 = tf.nn.dropout(fcl2, 0.5)        

        # 3rd FC 1000
        out = fully_connected(dr2, num_outputs=self.num_classes, activation_fn=None)

        return out