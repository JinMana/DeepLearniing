#coding=utf-8
import tempfile  #?
import tensorflow as tf
import pandas as np
from  resnet_utils import *

class hand_classifier(object):
    def __init__(self, model_save_path="./model_saving/hand_classifier"):
        self.model_save_path = model_save_path
    def identity_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training):
        block_name = 'res' + str(stage)+block    #这命名最后是hi什么形式的
        #out_filters 为输出通道
        f1, f2, f3 = out_filters
        with tf.variable_scope(block_name):
            X_shortcut = X_input   #shortcut部分
            #256, [64,64,256]

            #first,weight_variable和他的一个属性吗
            #1*1卷积
            W_conv1 = self.weight_variable([1,1,in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #three    15
            W_conv3 = self.weight_variable([kernel_size, kernel_size, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1,1,1,1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #final step
            add = tf.add(X, X_shortcut)   #?
            add_result = tf.nn.relu(add)   #see this，add and then relu

            return add_result

    def convolutional_block(self, X_input, kernel_size, in_filter, out_filters, stage, block, training, stride=2):
        block_name = 'res'+str(stage)+block
        with tf.variable_scope(block_name):
            f1, f2, f3 = out_filters
            X_shortcut = X_input

            #first
            W_conv1 = self.weight_variable([1,1,in_filter, f1])
            X = tf.nn.conv2d(X_input, W_conv1, strides=[1,stride, stride, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #second
            W_conv2 = self.weight_variable([kernel_size, kernel_size, f1, f2])
            X = tf.nn.conv2d(X, W_conv2, strides=[1, 1, 1, 1], padding='SAME')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)

            #three
            W_conv3 = self.weight_variable([1, 1, f2, f3])
            X = tf.nn.conv2d(X, W_conv3, strides=[1, 1, 1, 1], padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)

            #shortcut
            W_shortcut = self.weight_variable([1,1,in_filter, f3])
            X_shortcut = tf.nn.conv2d(X_shortcut, W_shortcut, strides=[1, stride, stride, 1], padding='VALID')

            #final
            add = tf.add(X, X_shortcut)
            add_result = tf.nn.relu(add)

            return add_result

    def deepnn(self, X_input, classes = 6):
        '''
        CONV2D - BN - RELU - MAXPOOL - CONVBLOCK - IDBLOCK*2 - CONVBLOCK - IDBLOCK*3
        -CONVBLOCK - IDBLOCK*5 - CONVBLOCK - IDBLOCK -AVGPOOL - PLAYER
        :param X_input:
        :param classes:
        :return:
        '''
        X = tf.pad(X_input, tf.constant([[0,0],[3,3, ],[3, 3],[0,0]]), 'CONSTANT')
        with tf.variable_scope('reference'):
            training = tf.placeholder(tf.bool, name='training')   #true false

            #stage1
            W_conv1 = self.weight_variable([7,7,3,64])
            X = tf.nn.conv2d(X, W_conv1, strides=[1,2,2,1],padding='VALID')
            X = tf.layers.batch_normalization(X, axis=3, training=training)
            X = tf.nn.relu(X)
            X = tf.nn.max_pool(X, ksize=[1,3,3,1], strides=[1,2,2,1], padding='VALID')

            assert (X.get_shape() == (X.get_shape()[0], 15, 15, 64))

            #stage2  x_input [? 15, 15, 64]
            #X_input, kernel_size, in_filter, out_filters, stage, block, train
            X = self.convolutional_block(X, 3, 64, [64, 64, 256],2, 'a', training, stride=1)
            # X_input, kernel_size, in_filter, out_filters, stage, block, training
            X = self.identity_block(X, 3, 256, [64,64,256], stage=2, block='b', training=training)
            X = self.identity_block(X, 3, 256, [64, 64, 256], stage=2, block='c', training=training)

            #stage3
            X = self.convolutional_block(X, 3, 256, [128,128,512], 3,'a', training=training)
            X = self.identity_block(X, 3, 512, [128, 128, 512], 3, 'b', training=training)
            X = self.identity_block(X, 3, 512, [128, 128, 512], 3, 'c', training=training)
            X = self.identity_block(X, 3, 512, [128, 128, 512], 3, 'd', training=training)

            #stage4
            X = self.convolutional_block(X, 3, 512, [256, 256, 1024],4, 'a', training=training)
            X = self.identity_block(X, 3, 1024, [256, 256, 1024], 4, 'b', training=training)
            X = self.identity_block(X, 3, 1024, [256, 256, 1024], 4, 'c', training=training)
            X = self.identity_block(X, 3, 1024, [256, 256, 1024], 4, 'd', training=training)
            X = self.identity_block(X, 3, 1024, [256, 256, 1024], 4, 'e', training=training)
            X = self.identity_block(X, 3, 1024, [256, 256, 1024], 4, 'f', training=training)

            #stage5
            X = self.convolutional_block(X, 3, 1024, [512, 512, 2048], 5, 'a', training=training)
            X = self.identity_block(X, 3, 2048, [512, 512,2048], 5, 'b', training=training)
            X = self.identity_block(X, 3, 2048, [512, 512,2048], 5, 'c', training=training)

            X = tf.nn.avg_pool(X, [1,2,2,1], strides=[1,1,1,1], padding='VALID')
            flatten = tf.layers.flatten(X)
            #dense
            X = tf.layers.dense(flatten, units=50, activation=tf.nn.relu)

            with tf.name_scope('dropout'):
                keep_prob = tf.placeholder(tf.float32)
                X = tf.nn.dropout(X, keep_prob)

            logits = tf.layers.dense(X, units=6, activation=tf.nn.relu)   #最后分成6类
            return logits, keep_prob, training
    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x)
    def weight_variable(self,shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    def cost(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)
        cross_entropy_cost = tf.reduce_mean(cross_entropy)
        return cross_entropy_cost
    def accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            corr = tf.equal(tf.argmax(logits, 1), tf.argmax(labels,1))
            corr_pre = tf.cast(corr, tf.float32)
        accuracy_op = tf.reduce_mean(corr_pre)
        return accuracy_op
    def train(self, x_train, y_train):
        features = tf.placeholder(tf.float32, [None, 64,64,3])
        labels = tf.placeholder(tf.int64, [None, 6])

        logits, keep_pro, train_model = self.deepnn(features)
        cross_entropy = self.cost(logits, labels)

        with tf.name_scope('adam_optimizer'):
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   #tf.GraphKeys.UPDATE_OPS保存训练之前完成的操作
            with tf.control_dependencies(update_op):      #这里设置了要run train_step也会运行update_op
                train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        graph_location = tempfile.mkdtemp()
        print('saving graph to:%s'%graph_location)

        train_writer = tf.summary.FileWriter(graph_location)
        train_writer.add_graph(tf.get_default_graph())

        mini_batches = random_mini_batches(x_train, y_train, mini_batch_size=32, seed=None)
        saver = tf.train.Saver()
        with tf.Session()  as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1000):
                x_mini_batch, y_mini_batch = mini_batches[np.random.randint(0, len(mini_batches))]
                train_step.run(feed_dict={features:x_mini_batch, labels:y_mini_batch, keep_pro:0.5, train_model:True})

                if i%20 == 0:
                    train_cost = sess.run(cross_entropy, feed_dict={features:x_mini_batch, labels:y_mini_batch, keep_pro:1.0, train_model:False})
                    print('step %d, training cost %g'%(i, train_cost))

            saver.save(sess, self.model_save_path)

    def evaluate(self, test_feature, test_labels, name='test'):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, [None, 64, 64, 3])
        y = tf.placeholder(tf.float32, [None, 6])

        logits, keep_prob, train_model = self.deepnn(x)
        accuracy = self.accuracy(logits, y)
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, self.model_save_path)
            accu = sess.run(accuracy, feed_dict={x:test_feature, y:test_labels, keep_prob:1.0, train_model:False})
            print('%s accuracy %g'%(name, accu))

def main(_):
    data_dir = './resnet50_dataset'
    orig_data = load_dataset(data_dir)
    x_train, y_train, x_test, y_test = process_orig_datasets(orig_data)

    model = hand_classifier()
    model.train(x_train, y_train)
    model.evaluate(x_test, y_test)
    model.evaluate(x_train, y_train, 'training data')

if __name__ == '__main__':
    tf.app.run(main=main)



