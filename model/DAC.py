import random
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DAC(object):
    def __ini__(self, clustered_object, data_size,
                      num_clusters=10, u_th=.95, l_th=.45, lambda_rate=.001,
                      batch_size=128, lr=.001,
                      step_size=20, save=None, restore=None):
        self._eps = 1e-10                               # For numerical stability
        self._clustered_object = clustered_object       # Data to be clustered
        self._data_size = data_size                     # Size of the input data
        self._num_clusters = num_clusters               # Number of clusters to divide
        self._u_th = u_th                               # Upper threshold for similary determination
        self._l_th = l_th                               # Lower threshold for dissimilary determination
        self._lambda_rate = lambda_rate                 # The rate which will change values of thresholds
        self._batch_size = batch_size                   # Size of each batch supplying to the NN
        self._lr = lr                                   # Learning rate
        self._step = step_size                          # Number of steps to print out training progress
        self._save = save                               # Path to save the model
        self._restore = restore                         # Path to restore the model
        self._saver = None                              # Saver of the model

        self._sess = None                               # Store object of tensorflow
        self.__init_graph()
        
    def __init_graph(self):
        # Computational graph
        self._tf_input_data = tf.placeholder(shape=[None, self._clustered_object.get_input_dim()], 
                                    dtype=tf.float32, name='input_data')
        self._tf_u_th = tf.placeholder(shape=[], dtype=tf.float32, name='u_th')
        self._tf_l_th = tf.placeholder(shape=[], dtype=tf.float32, name='l_th')
        self._tf_lr = tf.placeholder(shape=[], dtype=tf.float32, name='learning_rate')

        # Compute similarity matrix
        label_prob = self._clustered_object.get_model()
        label_prob_norm = tf.nn.l2_normalize(label_prob, dim=1)
        similarity_matrix = tf.matmul(label_prob_norm, label_prob_norm, transpose_b=True, name='similarity_matrix')

        # Determine similarity and disimilarity
        match_loc = tf.greater_equal(similarity_matrix, self._tf_u_th, name='match_location')
        unmatch_lo = tf.less(similarity_matrix, self._tf_l_th, name='unmatch_location')
        
        # Mask similarity and disimilary to numbers
        match_loc_mask = tf.cast(match_loc, dtype=tf.float32)
        unmatch_lo_mask = tf.cast(unmatch_lo, dtype=tf.float32)

        # Determine clusters
        self._tf_pred_label = tf.argmax(label_prob, axis=1)

        # Define loss and training operators
        match_entropy = tf.multiply(-tf.log(tf.clip_by_value(similarity_matrix, self._eps, 1.0)), match_loc_mask)
        unmatch_entropy = tf.multiply(-tf.log(tf.clip_by_value(1-similarity_matrix, self._eps, 1.0)), unmatch_lo_mask)

        self._tf_loss_sum = tf.reduce_mean(match_entropy) + tf.reduce_mean(unmatch_entropy)
        self._tf_train_op = tf.train.RMSPropOptimizer(self._tf_lr).minimize(self._tf_loss_sum)

        init_op = tf.initialize_all_variables()
        self._sess.run(init_op)

    def _save_model(self, global_step):
        if self._saver is None:
            self._saver = tf.train.Saver()
        
        if self._save is not None:
            self._saver.save(self._sess, save_path=self._save, global_step=global_step)

    def train(self, data):
        step = 0
        while self._u_th > self._l_th:
            step += 1
            batch_data, _ = self._get_data(data, self._batch_size)
            feed_dict = {
                self._tf_input_data: batch_data,
                self._tf_u_th: self._u_th,
                self._tf_l_th: self._l_th,
                self._tf_lr: self._lr
            }
            self._sess.run([self._tf_input_data, self._tf_train_op], feed_dict=feed_dict)
            
            # Check accuracy of the model
            if step % self._step == 0:
                batch_data, batch_label = self._get_data(data, self._batch_size * 2)
                feed_dict = {self._tf_input_data: batch_data}
                pred_cluster = self._sess.run(self._tf_pred_label, feed_dict=feed_dict)
                acc = self._get_accuracy(pred_cluster, batch_label)
                print('At step %d, acc = %f' % (step, acc))

                self._save_model(step)

            # Update thresholds
            self._u_th -= self._lambda_rate
            self._l_th += self._lambda_rate

    def _get_data(self, data, size):
        num_inputs, num_cols = data.shape[0], data.shape[1]
        batch_index = random.sample(num_inputs, size)
        
        # The last column of data contains labels
        batch_data = np.empty([size, num_cols-1], dtype=np.float32) 
        batch_label = np.empty([size], dtype=np.int32)

        # Sample input data
        for n, i in enumerate(batch_index):
            batch_data[n, ...] = data[i, :-1]
            batch_label[n] = data[-1]
        
        return batch_data, batch_label

    def _get_accuracy(self, y_pred, y_true):
        num_inputs = y_true.shape[0]
        acc_matrix = y_pred * y_true
        acc_matrix[acc_matrix > 0] = 1
        acc = np.sum(acc_matrix) / num_inputs

        return acc