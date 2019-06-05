
import model
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
import input_data
 
def get_one_image(train):
    '''Randomly pick one image from training data
    Return: ndarray
    '''
    n = len(train)
    ind = np.random.randint(0, n)
    img_dir = train[ind]
 
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image
 
 
def get_one_img(img_dir):
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([208, 208])
    image = np.array(image)
    return image
    
def evaluate_one_image():
    '''Test one image against the saved models and parameters
    '''
    train, train_label = input_data.get_files(train_dir)
    image_array = get_one_image(train)
 
    
    with tf.Graph().as_default():
        BATCH_SIZE = 1
        N_CLASSES = 2
        
        image = tf.cast(image_array, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1, 208, 208, 3])
        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        
        logit = tf.nn.softmax(logit)
        
        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
        
        # you need to change the directories to yours.
        logs_train_dir = '/home/hjxu/PycharmProjects/01_cats_vs_dogs/logs/recordstrain/' 
                       
        saver = tf.train.Saver()
        
        with tf.Session() as sess:
            
            print("Reading checkpoints...")
            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success, global_step is %s' % global_step)
            else:
                print('No checkpoint file found')
            
            prediction = sess.run(logit, feed_dict={x: image_array})
            max_index = np.argmax(prediction)
            if max_index==0:
                print('This is a cat with possibility %.6f' %prediction[:, 0])
            else:
                print('This is a dog with possibility %.6f' %prediction[:, 1])
