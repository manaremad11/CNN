from tensorflow.keras.datasets import mnist
import tensorflow as tf
import numpy as np

(train_x, train_y), (test_X, test_y) = mnist.load_data()

def normalization(x_data):
    max_value = np.max(x_data)
    min_value = np.min(x_data)
    diff = max_value - min_value
    result = np.subtract(x_data, min_value)
    result = np.divide(result, diff)
    return result

def weight_variable(shape):
    intial = tf.truncated_normal(shape,stddev=0.1)
    return tf.variable(intial)

def bias_variable(shape):
    intial = tf.constant(0.1,shape=shape)
    return tf.variable(intial)

W_Conv1 = weight_variable([5, 5, 3, 32])
B_Conv1 = bias_variable([32])
W_Conv2 = weight_variable([5, 5, 32, 64])
B_Conv2 = bias_variable([64])

def Max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

h_Conv1 = tf.nn.relu(conv2d(train_x, W_Conv1)+ B_Conv1)
h_pool1 = Max_pool(h_Conv1)
h_Conv2 = tf.nn.relu(conv2d(h_pool1, W_Conv2) + B_Conv2)
h_pool2 = Max_pool(h_Conv2)

w_FC1 = weight_variable([7*7*64*1024])
b_FC1 = bias_variable([1024])

keep_prob = tf.placeholder(tf.float32)
h_FC1_drop = tf.nn.dropout(b_FC1, keep_prob)

w_FC2 = weight_variable([1024*10])
b_FC2 = bias_variable([10])

Y_Conv = tf.matmul(h_FC1_drop, w_FC2) + b_FC2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(test_y, train_y))

train_step = tf.train.Adamoptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(test_y, 1), tf.argmax(train_y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


