import argparse
import sys
import pickle
import numpy as np

import cifar10_utils
import cifar100_utils

import tensorflow as tf
from tensorflow.contrib.layers import conv2d
from tensorflow.contrib.layers import max_pool2d
from tensorflow.contrib.layers import flatten
from tensorflow.contrib.layers import fully_connected

cifar10_dataset_folder_path = 'cifar-10-batches-py'
cifar100_dataset_folder_path = 'cifar-100-python'
save_model_path = './image_classification'

class VGG:
    def __init__(self, learning_rate, dataset='cifar10', model_type='A'):
        self.dataset = dataset
        if dataset == 'cifar10':
            self.num_classes = 10
        else:
            self.num_classes = 100

        self.learning_rate = learning_rate

        self.input = tf.placeholder(tf.float32, [None, 224, 224, 3], name='input')
        self.label = tf.placeholder(tf.int32, [None, self.num_classes], name='label')

        self.logits = self.load_model()
        self.model = tf.identity(self.logits, name='logits')

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label))
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

        if model_type != 'A' and model_type == 'A-LRN':
            group_1 = conv2d(group_1, num_outputs=64,
                                kernel_size=[3,3], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        group_1 = max_pool2d(group_1, kernel_size=[2,2], stride=2)

        # LAYER GROUP #2
        group_2 = conv2d(group_1, num_outputs=128,
                            kernel_size=[3, 3], padding='SAME',
                            activation_fn=tf.nn.relu)

        if model_type != 'A' and model_type == 'A-LRN':
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
            group_3 = conv2d(group_3, num_outputs=256,
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'D' or model_type == 'E':
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
            group_4 = conv2d(group_4, num_outputs=512,
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'D' or model_type == 'E':
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
            group_5 = conv2d(group_5, num_outputs=512,
                                kernel_size=[1,1], stride=1, padding='SAME',
                                activation_fn=tf.nn.relu)

        if model_type == 'D' or model_type == 'E':
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

    def _train_cifar10(self, epoch, batch_i, batch_size, valid_set):
        tmpValidFeatures, valid_labels = valid_set

        count = 0
        total_loss = 0

        for batch_features, batch_labels in cifar10_utils.load_preprocess_training_batch(batch_i, batch_size):
            loss, _ = sess.run([self.cost, self.optimizer],
                                    feed_dict={self.input: batch_features,
                                                self.label: batch_labels})
            total_loss = total_loss + loss
            count = count + 1

        print('Epoch {:>2}, CIFAR-10 Batch {}: Loss Average {:.6f}  '.format(epoch + 1, batch_i, total_loss/count), end='')

        # calculate the mean accuracy over all validation dataset
        valid_acc = 0
        for batch_valid_features, batch_valid_labels in cifar10_utils.batch_features_labels(tmpValidFeatures, valid_labels, batch_size):
            valid_acc += sess.run(self.accuracy,
                                    feed_dict={self.input:batch_valid_features,
                                                self.label:batch_valid_labels})

        tmp_num = tmpValidFeatures.shape[0]/batch_size
        print('Validation Accuracy {:.6f}'.format(valid_acc/tmp_num))        

    def _train_cifar100(self, epoch, batch_size, valid_set):
        tmpValidFeatures, valid_labels = valid_set        

        count = 0
        total_loss = 0

        for batch_features, batch_labels in cifar100_utils.load_preprocess_training_batch(batch_size):
            loss, _ = sess.run([self.cost, self.optimizer],
                                    feed_dict={self.input: batch_features,
                                                self.label: batch_labels})
            total_loss = total_loss + loss
            count = count + 1

        print('Epoch {:>2}, CIFAR-100 : Loss Average {:.6f}  '.format(epoch + 1, total_loss/count), end='')

        # calculate the mean accuracy over all validation dataset
        valid_acc = 0
        for batch_valid_features, batch_valid_labels in cifar100_utils.batch_features_labels(tmpValidFeatures, valid_labels, batch_size):
            valid_acc += sess.run(self.accuracy,
                                    feed_dict={self.input:batch_valid_features,
                                                self.label:batch_valid_labels})

        tmp_num = tmpValidFeatures.shape[0]/batch_size
        print('Validation Accuracy {:.6f}'.format(valid_acc/tmp_num))            

    def train(self, epochs, batch_size, valid_set, save_model_path):
        with tf.Session() as sess:
            print('global_variables_initializer...')
            sess.run(tf.global_variables_initializer())

            print('starting training ... ')
            for epoch in range(epochs):
                
                if self.dataset == 'cifar10':
                    n_batches = 5

                    for batch_i in range(1, n_batches + 1):
                        self._train_cifar10(epoch, batch_i, batch_size, valid_set)

                else:
                    self._train_cifar100(epoch, batch_size, valid_set)

            # Save Model
            saver = tf.train.Saver()
            save_path = saver.save(sess, save_model_path)

    def test(self, image, save_model_path):
        resize_images = []
        loaded_graph = tf.Graph()

        with tf.Session(graph=loaded_graph) as sess:
            loader = tf.train.import_meta_graph(save_model_path + '.meta')
            loader.restore(sess, save_model_path)        

            loaded_x = loaded_graph.get_tensor_by_name('input:0')
            loaded_y = loaded_graph.get_tensor_by_name('label:0')
            loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
            loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')

            resize_image = skimage.transform.resize(image, (224, 224), mode='constant')
            resize_images.append(resize_image)

            predictions = sess.run(
                tf.nn.softmax(loaded_logits),
                feed_dict={loaded_x: tmpTestFeatures, loaded_y: random_test_labels})

            label_names = load_label_names()

            predictions_array = []
            pred_names = []
            
            for index, pred_value in enumerate(predictions[0]):
                tmp_pred_name = label_names[index]
                predictions_array.append({tmp_pred_name : pred_value})

            return predictions_array

def parse_args(args):
    parser = argparse.ArgumentParser(description='Script for running VGG Net')

    parser.add_argument('--dataset', help='imagenet, cifar10 or cifar100. cifar10 is the default', default='cifar10')
    parser.add_argument('--dataset-path', help='location where the dataset is present', default='none')
    parser.add_argument('--learning-rate', help='learning rate', default=0.0001)
    parser.add_argument('--model-type', help='model type: A, A-LRN, B, C, D, E', default='A')
    parser.add_argument('--epochs', default=10)
    parser.add_argument('--batch-size', default=64)

    return parser.parse_args(args)

def main():
    args = sys.argv[1:]
    args = parse_args(args)

    dataset = args.dataset
    dataset_path = args.dataset_path
    learning_rate = args.learning_rate
    model_type = args.model_type
    epochs = args.epochs
    batch_size = args.batch_size

    if dataset == 'cifar10':
        cifar10_utils.download(cifar10_dataset_folder_path)
    else:
        cifar100_utils.download(cifar100_dataset_folder_path)

    if dataset == 'cifar10':
        print('preprocess_and_save_data...')
        cifar10_utils.preprocess_and_save_data(cifar10_dataset_folder_path)

        print('load features and labels for valid dataset...')
        valid_features, valid_labels = pickle.load(open('cifar10_preprocess_validation.p', mode='rb'))

        print('converting valid images to fit into imagenet size...')
        tmpValidFeatures = cifar10_utils.convert_to_imagenet_size(valid_features[:1000])
    elif dataset == 'cifar100':
        print('preprocess_and_save_data...')
        cifar100_utils.preprocess_and_save_data(cifar100_dataset_folder_path)

        print('load features and labels for valid dataset...')
        valid_features, valid_labels = pickle.load(open('cifar100_preprocess_validation.p', mode='rb'))

        print('converting valid images to fit into imagenet size...')
        tmpValidFeatures = cifar100_utils.convert_to_imagenet_size(valid_features[:1000])        
    else:
        sys.exit(0)

    vggNet = VGG(dataset='dataset', learning_rate=learning_rate, model_type=model_type)
    vggNet.train(epochs, batch_size, (tmpValidFeatures, valid_labels), save_model_path)

if __name__ == "__main__":
    main()
