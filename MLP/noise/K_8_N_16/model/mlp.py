#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mail: levy_lv@hotmail.com
# Lyu Wei @ 2017-07-26

'''
This is a 3-layer MLP decoder.
'''

import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class MLPModel(tf.keras.Model):
    def __init__(self, length_input, dropout_keep_prob):
        super(MLPModel, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.dropout1 = tf.keras.layers.Dropout(1 - dropout_keep_prob)
        self.fc2 = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.dropout2 = tf.keras.layers.Dropout(1 - dropout_keep_prob)
        self.fc3 = tf.keras.layers.Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.dropout3 = tf.keras.layers.Dropout(1 - dropout_keep_prob)
        self.output_layer = tf.keras.layers.Dense(int(length_input / 2), activation=None)

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.dropout1(x, training=training)
        x = self.fc2(x)
        x = self.dropout2(x, training=training)
        x = self.fc3(x)
        x = self.dropout3(x, training=training)
        return self.output_layer(x)


def get_random_batch_data(x, y, batch_size):
    '''get random batch data from x and y, which have the same length'''
    index = np.random.randint(0, len(x) - batch_size)
    return x[index:(index + batch_size)], y[index:(index + batch_size)]


# Parameters setting
N = 16
K = 8
data_path = '../../../../data/noise/K_8_N_16/'
num_epoch = 10 ** 5
batch_size = 128
train_snr = np.arange(-2, 22, 2)
test_snr = np.arange(0, 6.5, 0.5)
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
epoch_setting = np.array([10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])
res_ber = np.zeros([len(train_ratio), len(train_snr), len(test_snr), len(epoch_setting)])

# Make the MLP model
dropout_keep_prob = 0.9
model = MLPModel(N, dropout_keep_prob)
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training and Testing Loop
backward_batch_time_total = 0.0
forward_time_total = 0.0
for ratio_index in range(len(train_ratio)):
    for tr_snr_index in range(len(train_snr)):
        train_filename = f'ratio_{train_ratio[ratio_index]}_train_snr_{train_snr[tr_snr_index]}dB.mat'
        train_data = sio.loadmat(data_path + train_filename)
        x_train = train_data['x_train']
        y_train = train_data['y_train']

        print('---------------------------------')
        print('New beginning')
        print('---------------------------------')
        for epoch in range(num_epoch):
            x_batch, y_batch = get_random_batch_data(x_train, y_train, batch_size)

            # Training
            backward_batch_time_start = time.time()
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)
                total_loss = loss + sum(model.losses)  # Adding regularization losses
            grads = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            duration = time.time() - backward_batch_time_start
            backward_batch_time_total += duration

            if (epoch + 1) % 1000 == 0:
                prediction = tf.cast(logits > 0.5, tf.float32)
                correct_prediction = tf.equal(prediction, y_batch)
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                train_ber = 1.0 - accuracy
                print(
                    f'{datetime.now()}: epoch = {epoch + 1}, train_ratio = {train_ratio[ratio_index]}, train_snr = {train_snr[tr_snr_index]} dB ---> train_ber = {train_ber:.4f}, forward time = {duration:.3f} sec')

            if epoch + 1 in epoch_setting:
                epoch_index = np.where(epoch_setting == (epoch + 1))[0][0]

                print('\n***********TEST BEGIN************')
                for te_snr_index in range(len(test_snr)):
                    test_filename = f'test_snr_{test_snr[te_snr_index]}dB.mat'
                    test_data = sio.loadmat(data_path + test_filename)
                    x_test = test_data['x_test']
                    y_test = test_data['y_test']

                    # Testing
                    forward_time_start = time.time()
                    logits = model(x_test, training=False)
                    prediction = tf.cast(logits > 0.5, tf.float32)
                    correct_prediction = tf.equal(prediction, y_test)
                    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    test_ber = 1.0 - accuracy
                    duration = time.time() - forward_time_start
                    forward_time_total += duration
                    print(
                        f'{datetime.now()}: epoch = {epoch + 1}, train_ratio = {train_ratio[ratio_index]}, train_snr = {train_snr[tr_snr_index]} dB, test_snr = {test_snr[te_snr_index]} dB ---> ber = {test_ber:.4f}, forward time = {duration:.3f} sec')
                    res_ber[ratio_index, tr_snr_index, te_snr_index, epoch_index] = test_ber
                print('***********TEST END**********\n')

# Statistics
backward_batch_time_avg = backward_batch_time_total / (num_epoch * len(train_ratio) * len(train_snr))
forward_count = int(len(epoch_setting) * len(test_snr) * len(train_ratio) * len(train_snr))
forward_time_avg = forward_time_total / forward_count

print('\n*****************END***************')
print(
    f'For the backward time of MLP:\nbatch size = {batch_size}, total batch = {num_epoch}, time = {backward_batch_time_avg:.3f} sec/batch')
print(
    f'For the forward time of MLP:\ntest number = {len(x_test)}, total test = {forward_count}, time = {forward_time_avg:.3f} sec')
sio.savemat('../../../../Results/Result-with-noise/MLP_Result/K_8_N_16/mlp_result',
            {'ber_trainRatio_trainSNR_testSNR_epoch': res_ber, 'backward_batch_time_avg': backward_batch_time_avg,
             'forward_time_avg': forward_time_avg})
