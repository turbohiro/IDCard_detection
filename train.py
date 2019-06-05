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

N_CLASSES = 3
IMG_W = 500  # resize the image, if the input image is too large, training will be very slow.
IMG_H = 500
BATCH_SIZE = 16
CAPACITY = 2000
MAX_STEP = 10000 # with current parameters, it is suggested to use MAX_STEP>10k
learning_rate = 0.0001 # with current parameters, it is suggested to use learning rate<0.0001


def run_training1():
    
    logs_train_dir = dataDir + '/logs/recordstrain/'
#    
#    train, train_label = input_data.get_files(train_dir)
    tfrecords_file = 'Idcard_train.tfrecords'
    train_batch, train_label_batch = read_and_decode(tfrecords_file, batch_size=BATCH_SIZE)
    train_batch = tf.cast(train_batch,dtype=tf.float32)
    train_label_batch = tf.cast(train_label_batch,dtype=tf.int64)
    
    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)        
    train_op = model.trainning(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)
       
    summary_op = tf.summary.merge_all()
    sess = tf.Session()
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    try:
        for step in np.arange(MAX_STEP):
            if coord.should_stop():
                    break
            _, tra_loss, tra_acc = sess.run([train_op, train_loss, train__acc])
               
            if step % 50 == 0:
                print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
                summary_str = sess.run(summary_op)
                train_writer.add_summary(summary_str, step)
            
            if step % 2000 == 0 or (step + 1) == MAX_STEP:
                checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                
    except tf.errors.OutOfRangeError:
        print('Done training -- epoch limit reached')
    finally:
        coord.request_stop()
        
    coord.join(threads)
    sess.close()
    
run_training1()
