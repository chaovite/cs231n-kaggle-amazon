########################################################################################
# VGG16 trainable implementation in TensorFlow                                         #
# Modified from Davi Frossard, 2016                                                    #
# Details:                                                                             #
# http://www.cs.toronto.edu/~frossard/post/vgg16/                                      #
#                                                                                      #
# Model from https://gist.github.com/ksimonyan/211839e770f7b538e2d8#file-readme-md     #
# Weights from Caffe converted using https://github.com/ethereon/caffe-tensorflow      #
########################################################################################

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import preprocess_utils import batch_data_generator_train
import time

    
class vgg16_trainable:
    """ A trainable vgg16 model that allow fine tuning the last fc layer."""

    def __init__(self, weights, sess, learning_rate = 1e-2, weight_scale = 1e-4,
                 height=224, width=224, channel=3, num_classes=17, x_mean = None):
        
        self.x  = tf.placeholder(tf.float32, shape=[None, height, width, channel])
        self.y  = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.lr = learning_rate
        self.height= height
        self.width = width
        self.weights = weights # weights path 
        self.weight_scale = weight_scale
        self.x_mean = x_mean
        
        # build neural network architecture.
        self.convlayers()
        self.fc_layers(weight_scale)
        
        # define loss function as binary entropy loss
        loss_el = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels = self.y,
                        logits = self.logits,
                        name='elementwise_loss')
        self.mean_loss = tf.divide(tf.reduce_sum(loss_el), tf.cast(tf.shape(loss_el)[0], tf.float32))
        
        # define optimizer and set learning rate
        self.learning_rate = learning_rate
        optimizer  = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.mean_loss)

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        else:
            raise ValueError('Must specify weights file path and sess')
        self.best_fs = -1
        self.best_model_path = None
    
    def load_weights(self, weight_file, sess):
        """Assign pretrained weights and biases to all layers from weight_file"""
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if not 'fc8' in k:
#                print(i, k, np.shape(weights[k]))
                sess.run(self.parameters[i].assign(weights[k]))
            else:
                print('%s is not assigned but initialized' % (k))

    def run_model(self, session, Xd, yd, X_val = None, y_val = None,
                  epochs=1, batch_size=64, print_every = 100, training = False, 
                  plot_losses = False, verbose=False, checkpoint_path = None):
        """method for training and evaluation """
        # convert logits to label prediction.
        y_out      =  logits2y(self.logits)
        
        accuracy, correct_prediction  = correct_tags(self.y, y_out)
        fs = multi_f_beta_score(self.y, y_out, 2.0)
        
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)
    
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self.mean_loss, correct_prediction, accuracy]
        
        if training:
            # if it's training, replace accuracy with training step.
            variables[-1] = self.train_step
    
        # counter 
        iter_cnt = 0
        self.validation_acc = []
        self.train_acc = []
        self.losses = []
        self.train_fscore = []
        self.val_fscore = []
        
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            total_loss = 0
            total_fs   = 0
            tic_e = time.time()
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                tic_b = time.time()
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]
                
                # generate batch of data with augmentation.
                x_batch, y_batch = batch_data_generator_train(Xd, yd, batch_size)
                
                # create a feed dictionary for this batch
                feed_dict = {self.x: x_batch,
                             self.y: y_batch}
            
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)
                fs_batch      = session.run(fs, feed_dict=feed_dict)
                total_fs     += fs_batch*batch_size

                # aggregate performance stats
                self.losses.append(loss*batch_size)
                total_loss += loss*batch_size
                correct += np.sum(corr)
                
                toc_b = time.time()
            
                # print every now and then
                if verbose:
                    if training and (iter_cnt % print_every) == 0:
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}, F2 score: {3:f}"\
                              .format(iter_cnt,loss,np.sum(corr)/batch_size, fs_batch))
                        print('Elapse time from last print %f' % (toc_b-tic_b))
                        
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = total_loss/Xd.shape[0]
            total_fs   = total_fs/Xd.shape[0]
            toc_e = time.time()

            print("Epoch {2}, Overall mean loss = {0:.3g} and training accuracy of {1:.3g}, F2 score: {3:f}"\
                  .format(total_loss,total_correct,e+1, total_fs))
            print('Elapse time from last epoch %f' % (toc_e-tic_e))
        
            # evaluate the evaluation accuracy.
            is_val = not ((X_val is None) or (y_val is None))
            self.train_acc.append(total_correct)
            self.train_fscore.append(total_fs)
            
            #saver = tf.train.Saver()
            if is_val:            
                # run evaluation batch by batch to avoid OOM
                acc_val = 0
                fs_val  = 0
                for i in range(int(math.ceil(X_val.shape[0]/batch_size))):
                    # generate indicies for the batch
                    start_idx = (i*batch_size)%X_val.shape[0]
                    # shuffle indicies
                    val_indicies = np.arange(X_val.shape[0])
                    np.random.shuffle(val_indicies)
                    idx = val_indicies[start_idx:start_idx+batch_size]

                    # create a feed dictionary for this batch
                    feed_dict_val = {self.x: X_val[idx,:],
                                     self.y: y_val[idx]}
                    # get batch size
                    actual_batch_size = y_val[i:i+batch_size].shape[0]
                    
                    acc_val_b, fs_val_b = session.run([accuracy, fs],feed_dict=feed_dict_val)
                    acc_val += acc_val_b*actual_batch_size
                    fs_val  +=  fs_val_b*actual_batch_size
            
                acc_val = acc_val/X_val.shape[0]
                fs_val = fs_val/X_val.shape[0]
                
                
                print("Epoch %d, validation accuracy: %f, F2 score: %f"\
                  % (e+1, acc_val, fs_val))
                self.validation_acc.append(acc_val)
                self.val_fscore.append(fs_val)
                
                if fs_val > self.best_fs:
                    self.best_fs = fs_val
                    if checkpoint_path:
                        save_path = saver.save(sess, checkpoint_path)
                        self.best_model_path = save_path
                        print('Best pretrained vgg model saved in %s' % (save_path))

            if plot_losses:
                plt.plot(self.losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()

    def predict_tag(self, session, X_test, threshold = 0.235, label_list = None, batch_size = 100):
            
        """This may not work when X_test is too big, consider breaking it into batches"""
        y_out      =  logits2y(self.logits, threshold= threshold)
        feed_dict_predict = {self.x: X_test,
                             self.y: None}
        y_test     =  session.run(y_out, feed_dict=feed_dict_predict)
        tag_test   = None
        if label_list:
            tag_test   =  y2tags(y_test, label_list)
        return y_test, tag_test
        
    def val_acc(self, session, X_val, y_val, threshold=0.235):
        """evaluate accuracy"""
        y_out      =  logits2y(self.logits, threshold=threshold)
        acc_v, corrects_v  = correct_tags(self.y, y_out)
        feed_dict = {self.x: X_val,
                     self.y: y_val}
        variables = [acc_v, corrects_v]
        acc,  corrects =  session.run(variables, feed_dict=feed_dict)
        return acc, corrects

    def convlayers(self):
        """Define convolutional layers"""
        self.parameters = []
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            if not self.x_mean is None:
                self.x_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='x_mean')
            images = self.x - self.x_mean
            
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                             initializer = tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                             trainable = False)
            #
                           
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable = False)
            
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                             initializer = tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),    
                             trainable = False)
                         
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                             initializer =tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                             initializer =tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),  
                              trainable = False)
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.variable_scope('conv3_1') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                             initializer =tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
        
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]
            
        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]
            

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
        
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self, weight_scale):
        """Define fully connected layers"""
        # fc1
        with tf.variable_scope('fc6') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            
            fc1w = tf.get_variable('weights', dtype= tf.float32,
                                initializer = tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=weight_scale),
                                trainable = True)
            fc1b = tf.get_variable('biases', dtype= tf.float32,
                                initializer = tf.constant(weight_scale, shape=[4096], dtype=tf.float32),
                                trainable = True)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.variable_scope('fc7') as scope:
        
            fc2w = tf.get_variable('weights', dtype= tf.float32,
                                initializer = tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=weight_scale),
                                trainable = True)
            fc2b = tf.get_variable('biases', dtype= tf.float32,
                                initializer = tf.constant(weight_scale, shape=[4096], dtype=tf.float32),
                                trainable = True)             
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3 # freeze the weights and bias of all the other layers and only train this fully connected layer.
        with tf.variable_scope('fc8') as scope:
        
            fc3w = tf.get_variable('weights', dtype= tf.float32,
                                initializer = tf.truncated_normal([4096, 17], dtype=tf.float32, stddev = weight_scale),
                                trainable = True)
            fc3b = tf.get_variable('biases', dtype= tf.float32,
                                initializer = tf.constant(weight_scale, shape=[17], dtype=tf.float32),
                                trainable = True) 
            
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.logits = self.fc3l
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        """Assign pretrained weights and biases to all layers from weight_file"""
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            if not 'fc8' in k:
                sess.run(self.parameters[i].assign(weights[k]))
            else:
                print('%s is not assigned but initialized' % (k))
        
        vars_train = tf.trainable_variables()
        names = []
        for var in vars_train:
            names.append(var.name)
        print('trainable variables: ', names)
                
    
# some utility functions
def logits2y(logits, threshold=0.235):
    """convert logitst to predictions"""
    p = tf.sigmoid(logits)
    return tf.cast(tf.less(threshold, p), tf.int32)

def correct_tags(y, y_out):
    """compare y and yout and output the accuracy"""
    y = tf.cast(y, tf.float32)
    y_out = tf.cast(y_out, tf.float32)
    n = tf.cast(tf.shape(y)[0],tf.float32)
    wrongs  = tf.reduce_sum(tf.reduce_max(tf.abs(tf.subtract(y, y_out)), 1))
    corrects = tf.subtract(n, wrongs)
    acc     = tf.divide(corrects, n)
    return acc, corrects

def multi_f_beta_score(y, y_out, beta=2.0):
    """compute mean F_beta score"""
    eps = 1e-6
    y = tf.cast(y, tf.float32)
    y_out = tf.cast(y_out, tf.float32)
    num_pos = tf.reduce_sum(y_out, axis=1)
    tp      = tf.reduce_sum(y_out*y,axis=1)
    num_pos_hat = tf.reduce_sum(y, axis=1)
    precision = tp/(num_pos+eps)
    recall    = tp/(num_pos_hat+eps)
    fs = (1+beta*beta)*precision*recall/(beta*beta*precision+recall+eps)
    f  = tf.reduce_mean(fs,axis=0)
    return f
                
def y2tags(ys, label_list):
    """
    Use label_list to convert one-hot vector y into tags
    """
    dim = ys.shape
    if len(dim)==1:
        n=1
        m=dim[0]
    else:
        n,m = dim
    tags = []
    for i in range(n):
        if n==1:
            y = ys
        else:
            y   = ys[i]
        tag = ''
        for j in range(m):
            if y[j] == 1:
                tag += label_list[j] + ' '
        tag=tag.strip()
        tags.append(tag)
    return tags
