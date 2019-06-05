import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import glob
import seaborn as sns
from PIL import Image
import glob
import tensorflow as tf
import keras
from keras import backend as K
from keras import layers
from keras.callbacks import Callback
os.environ['CUDA_VISIBLE_DEVICES']='1'

dataDir = '/data/jupyter/libin713/sample_IDCard'

def read_and_decode(filename,batch_size):  # 读入dog_train.tfrecords
    filename_queue = tf.train.string_input_producer([filename])  # 生成一个queue队列
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # 返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # 将image数据和label取出来
 
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [500, 500, 3])  # reshape为128*128的3通道图片
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5  # 在流中抛出img张量
    label = tf.cast(features['label'], tf.int32)  # 在流中抛出label张量
    img_batch,label_batch = tf.train.batch([img,label],batch_size = batch_size,num_threads = 64,capacity = 2000)
    return img_batch,tf.reshape(label_batch,[batch_size])

sess = K.get_session()
batch_size = 2
batch_shape = (batch_size, 500, 500, 3)
epochs = 1000
num_classes = 3
BATCH_SIZE = 2
tfrecords_file = 'Idcard_train.tfrecords'
train_batch, train_label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
train_batch = tf.cast(train_batch,dtype=tf.float32)
train_label_batch = tf.cast(train_label_batch,dtype=tf.int64)

y_train_batch = tf.one_hot(train_label_batch, num_classes)

def cnn_layers(x_train_input):
    x = layers.Conv2D(32, (3, 3),
                      activation='relu', padding='valid')(x_train_input)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x_train_out = layers.Dense(num_classes,
                               activation='softmax',
                               name='x_train_out')(x)
    return x_train_out
 

x_batch_shape = train_batch.get_shape().as_list()
y_batch_shape = y_train_batch.get_shape().as_list()
 
model_input = layers.Input(tensor=train_batch)
model_output = cnn_layers(model_input)
train_model = keras.models.Model(inputs=model_input, outputs=model_output)
 
# Pass the target tensor `y_train_batch` to `compile`
# via the `target_tensors` keyword argument:
# 将目标张量“Y-y_train_batch”通过“target_tensors”关键字参数“编译”：
train_model.compile(optimizer=keras.optimizers.RMSprop(lr=2e-3, decay=1e-5),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'],
                    target_tensors=[y_train_batch])
train_model.summary()


 
train_model.fit(epochs=epochs,steps_per_epoch=100)
 
# Save the model weights.
# 保存模型权重
train_model.save_weights('saved_wt.h5')

 




