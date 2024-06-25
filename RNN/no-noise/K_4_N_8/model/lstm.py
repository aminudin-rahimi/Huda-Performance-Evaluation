#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mail:levy_lv@hotmail.com
# Lyu Wei @ 2017-07-26

'''
This is a LSTM decoder.
'''

import tensorflow as tf
import numpy as np
import scipy.io as sio
import math
import time
from datetime import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class LSTMModel(tf.keras.Model):
    def __init__(self, N, K, hidden_size, dropout_keep_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, return_state=False)
        self.dropout = tf.keras.layers.Dropout(1 - dropout_keep_prob)
        self.dense = tf.keras.layers.Dense(K, activation='sigmoid')

    def call(self, inputs, training=False):
        x = self.lstm(inputs)
        x = self.dropout(x, training=training)
        x = self.dense(x[:, -1, :])
        return x


def get_random_batch_data(x, y, batch_size):
    '''get random batch data from x and y, which have the same length
    '''
    index = np.random.randint(0, len(x) - batch_size)
    return x[index:(index + batch_size)], y[index:(index + batch_size)]


# Parameters setting
N = 8
K = 4
data_path = '../../../../data/no-noise/K_4_N_8/'
num_epoch = 10 ** 5
train_batch_size = 128
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
epoch_setting = np.array([10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])
res_ber = np.zeros([len(train_ratio), len(epoch_setting)])

# make the LSTM model
dropout_keep_prob = 0.8
model = LSTMModel(N, K, hidden_size=256, dropout_keep_prob=dropout_keep_prob)
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training and Testing Loop
backward_batch_time_total = 0.0
forward_time_total = 0.0
for ratio_index in range(len(train_ratio)):
    train_filename = 'ratio_' + str(train_ratio[ratio_index]) + '.mat'
    train_data = sio.loadmat(data_path + train_filename)
    x_train = train_data['x_train']
    y_train = train_data['y_train']

    print('---------------------------------')
    print('New beginning')
    print('---------------------------------')
    for epoch in range(num_epoch):
        x_batch, y_batch = get_random_batch_data(x_train, y_train, train_batch_size)

        # Training
        backward_batch_time_start = time.time()
        with tf.GradientTape() as tape:
            logits = model(tf.expand_dims(x_batch, -1), training=True)
            loss = loss_fn(y_batch, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        duration = time.time() - backward_batch_time_start
        backward_batch_time_total += duration

        if (epoch + 1) % 1000 == 0:
            train_ber = 1 - tf.reduce_mean(tf.cast(tf.equal(tf.cast(logits > 0.5, tf.float32), y_batch), tf.float32))
            print(
                f'{datetime.now()}: epoch = {epoch + 1}, train_ratio = {train_ratio[ratio_index]} ---> train_ber = {train_ber:.4f}, forward time = {duration:.3f} sec')

        if epoch + 1 in epoch_setting:
            epoch_index = int(np.log10(epoch + 1) - 1)

            print('\n***********TEST BEGIN************')
            test_filename = 'test.mat'
            test_data = sio.loadmat(data_path + test_filename)
            x_test = test_data['x_test']
            y_test = test_data['y_test']

            # Testing
            forward_time_start = time.time()
            logits = model(tf.expand_dims(x_test, -1), training=False)
            test_ber = 1 - tf.reduce_mean(tf.cast(tf.equal(tf.cast(logits > 0.5, tf.float32), y_test), tf.float32))
            duration = time.time() - forward_time_start
            forward_time_total += duration
            print(
                f'{datetime.now()}: epoch = {epoch + 1}, train_ratio = {train_ratio[ratio_index]} ---> ber = {test_ber:.4f}, forward time = {duration:.3f} sec')
            res_ber[ratio_index, epoch_index] = test_ber
            print('***********TEST END**********\n')

# Statistics
backward_batch_time_avg = backward_batch_time_total / (num_epoch * len(train_ratio))
forward_count = len(epoch_setting) * len(train_ratio)
forward_time_avg = forward_time_total / forward_count

print('\n*****************END***************')
print(
    f'For the backward time of LSTM:\nbatch size = {train_batch_size}, total batch = {num_epoch}, time = {backward_batch_time_avg:.3f} sec/batch')
print(
    f'For the forward time of LSTM:\ntest number = {len(x_test)}, total test = {forward_count}, time = {forward_time_avg:.3f} sec')
sio.savemat('../../../../Results/LSTM_Result/K_4_N_8/lstm_result', {'ber_trainRatio_epoch': res_ber, 'backward_batch_time_avg': backward_batch_time_avg,
                                   'forward_time_avg': forward_time_avg})
