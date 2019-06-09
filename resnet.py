import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import seaborn as sns
from PIL import Image
import glob
import tensorflow as tf
import model
os.environ['CUDA_VISIBLE_DEVICES']='0'

dataDir = '/data/jupyter/libin713/sample_IDCard'

#1)读取tfrecords数据训练集
def read_and_decode(filename):  # 读入tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [600, 600, 3])  # reshape为600*600的3通道图
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    return img,label

tfrecords_file = 'Idcard_train1.tfrecords'
img_train,labels_train =read_and_decode(tfrecords_file)
train_batch_size = 1134
img_train_batch,labels_train_batch = tf.train.shuffle_batch([img_train,labels_train],batch_size = train_batch_size,
                                                            capacity = 1200,min_after_dequeue= 500, num_threads= 3)
classes = 3
train_labels = tf.one_hot(labels_train_batch,classes,1,0)

tfrecords_file_test = 'Idcard_test1.tfrecords'
img_test,labels_test =read_and_decode(tfrecords_file_test)
test_batch_size = 201
img_test_batch,labels_test_batch = tf.train.shuffle_batch([img_test,labels_test],batch_size = test_batch_size,
                                                            capacity = 1000,min_after_dequeue= 500, num_threads= 3)
classes = 3
test_labels = tf.one_hot(labels_test_batch,classes,1,0)

print(img_train_batch,train_labels)
print(img_test_batch,test_labels)

import tflearn
n = 5
# Real-time data preprocessing
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_crop([600, 600], padding=4)

# Building Residual Network
net = tflearn.input_data(shape=[None, 600, 600, 3],
                         data_preprocessing=img_prep,
                         data_augmentation=img_aug)
net = tflearn.conv_2d(net, 16, 3, regularizer='L2', weight_decay=0.001)
net = tflearn.residual_block(net, n, 16)
net = tflearn.residual_block(net, 1, 32, downsample=True)
net = tflearn.residual_block(net, n-1, 32)
net = tflearn.dropout(net, 0.5) 
net = tflearn.residual_block(net, 1, 64, downsample=True)
net = tflearn.residual_block(net, n-1, 64)
net = tflearn.batch_normalization(net)
net = tflearn.activation(net, 'relu')
net = tflearn.global_avg_pool(net)
# Regression
net = tflearn.fully_connected(net, 3, activation='softmax')
mom = tflearn.Momentum(0.001, lr_decay=0.001, decay_step=32000, staircase=True)
net = tflearn.regression(net, optimizer=mom,
                         loss='categorical_crossentropy')
# Training
model = tflearn.DNN(net, checkpoint_path='/data/jupyter/libin713/sample_IDCard/model_resnet_Idcard/resnet_Idcard.model',
                    max_checkpoints=10, tensorboard_verbose=0,
                    clip_gradients=0.)

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(init)
sess2.run(init)
coord1 = tf.train.Coordinator()
coord2 = tf.train.Coordinator()
threads1 = tf.train.start_queue_runners(sess = sess1,coord = coord1)
threads2 = tf.train.start_queue_runners(sess = sess2,coord = coord2)
#for i in range(2000):
X,Y = sess1.run([img_train_batch,train_labels])
X_test,Y_test = sess2.run([img_test_batch,test_labels])
print(X.shape,X_test.shape)
model.fit(X, Y, n_epoch=100, validation_set=(X_test,Y_test),
      snapshot_epoch=False, snapshot_step=500,
      show_metric=True, batch_size=4, shuffle=True,
      run_id='resnet_Idcard')
model.save("resnet_Idcard.model")
