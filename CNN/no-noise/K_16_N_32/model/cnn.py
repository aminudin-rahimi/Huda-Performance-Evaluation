#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# mail: levy_lv@hotmail.com
# Lyu Wei @ 2017-07-24

'''
This is a CNN decoder.
'''

import tensorflow as tf
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CNNModel(tf.keras.Model):
    def __init__(self, length_input, dropout_keep_prob):
        super(CNNModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(8, (1, 3), activation='relu', padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool1 = tf.keras.layers.AvgPool2D((1, 2), strides=(1, 2), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(16, (1, 3), activation='relu', padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool2 = tf.keras.layers.AvgPool2D((1, 2), strides=(1, 2), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(32, (1, 3), activation='relu', padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.0005))
        self.pool3 = tf.keras.layers.AvgPool2D((1, 2), strides=(1, 2), padding='same')
        self.fc4 = tf.keras.layers.Conv2D(int(length_input / 2), (1, int(length_input / 8)), activation=None,
                                          padding='valid')
        self.dropout = tf.keras.layers.Dropout(1 - dropout_keep_prob)

    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.fc4(x)
        if training:
            x = self.dropout(x, training=training)
        return tf.squeeze(x, [1, 2])


def get_random_batch_data(x, y, batch_size):
    '''get random batch data from x and y, which have the same length'''
    index = np.random.randint(0, len(x) - batch_size)
    return x[index:(index + batch_size)], y[index:(index + batch_size)]


# Parameters setting
N = 32
K = 16
data_path = '../../../../data/no-noise/K_16_N_32/'
num_epoch = 10 ** 5
batch_size = 128
train_ratio = np.array([0.4, 0.6, 0.8, 1.0])
epoch_setting = np.array([10, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5])
res_ber = np.zeros([len(train_ratio), len(epoch_setting)])

# Make the CNN model
dropout_keep_prob = 0.9
model = CNNModel(N, dropout_keep_prob)
optimizer = tf.keras.optimizers.Adam(1e-3)
loss_fn = tf.keras.losses.MeanSquaredError()

# Training and Testing Loop
backward_batch_time_total = 0.0
forward_time_total = 0.0
for ratio_index in range(len(train_ratio)):
    train_filename = f'ratio_{train_ratio[ratio_index]}.mat'
    train_data = sio.loadmat(data_path + train_filename)
    x_train = train_data['x_train']
    y_train = train_data['y_train']

    print('---------------------------------')
    print('New beginning')
    print('---------------------------------')
    for epoch in range(num_epoch):
        x_batch, y_batch = get_random_batch_data(x_train, y_train, batch_size)
        x_batch = np.reshape(x_batch, [-1, 1, N, 1])

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
                f'{datetime.now()}: epoch = {epoch + 1}, train_ratio = {train_ratio[ratio_index]} ---> train_ber = {train_ber:.4f}, forward time = {duration:.3f} sec')

        if epoch + 1 in epoch_setting:
            epoch_index = int(np.log10(epoch + 1) - 1)

            print('\n***********TEST BEGIN************')
            test_filename = 'test.mat'
            test_data = sio.loadmat(data_path + test_filename)
            x_test = test_data['x_test']
            y_test = test_data['y_test']
            x_test = np.reshape(x_test, [-1, 1, N, 1])

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
                f'{datetime.now()}: epoch = {epoch + 1}, train_ratio = {train_ratio[ratio_index]} ---> ber = {test_ber:.4f}, forward time = {duration:.3f} sec')
            res_ber[ratio_index, epoch_index] = test_ber
            print('***********TEST END**********\n')

# Statistics
backward_batch_time_avg = backward_batch_time_total / (num_epoch * len(train_ratio))
forward_count = int(len(epoch_setting) * len(train_ratio))
forward_time_avg = forward_time_total / forward_count

print('\n*****************END***************')
print(
    f'For the backward time of CNN:\nbatch size = {batch_size}, total batch = {num_epoch}, time = {backward_batch_time_avg:.3f} sec/batch')
print(
    f'For the forward time of CNN:\ntest number = {len(x_test)}, total test = {forward_count}, time = {forward_time_avg:.3f} sec')
sio.savemat('../../../../Results/CNN_Result/K_16_N_32/cnn_result',
            {'ber_trainRatio_epoch': res_ber, 'backward_batch_time_avg': backward_batch_time_avg,
             'forward_time_avg': forward_time_avg})
