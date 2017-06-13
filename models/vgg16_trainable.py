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
import models.image_prep_helpers as helpers
import time


def conv_bn_relu(inputs, num_filters, kernel_size, strides, training):
    out = tf.layers.conv2d(inputs, num_filters, kernel_size, 
                           strides=strides, padding='same')
    out = tf.layers.batch_normalization(out, training=training)
    out = tf.nn.relu(out)
    return out

def fc_bn_relu(inputs, units, training):
    out = tf.layers.dense(inputs, units)
    out = tf.layers.batch_normalization(out, training=training)
    out = tf.nn.relu(out)
    return out

    
class vgg16_trainable:
    """ A trainable vgg16 model that allow fine tuning the last fc layer."""

    def __init__(self, weights, sess, learning_rate = 5e-5, weight_scale = 1e-4,
                 drop_out =0.5, lr_decay = 0.85, max_epoch = 5,
                 height=224, width=224, channel=3, num_classes=17, x_mean = None, 
                 x_std =None, weights_classes = None, loss_type='binary'):
        
        self.x  = tf.placeholder(tf.float32, shape=[None, height, width, channel])
        self.y  = tf.placeholder(tf.float32, shape=[None, num_classes])
        self.lr = tf.placeholder(tf.float32)
        self.istraining = tf.placeholder(tf.bool)
        self.num_classes = num_classes
        self.lr_var = learning_rate
        self.drop_out = drop_out
        self.lr_decay = lr_decay
        self.max_epoch = max_epoch
        self.height= height
        self.width = width
        self.weights = weights # weights path 
        self.weight_scale = weight_scale
        
        self.x_mean = x_mean
        self.x_std  = x_std 
        
        # build neural network architecture.
        self.convlayers()
        self.fc_layers()
        
        assert loss_type is in ['binary','binary_softmax']
        
        self.loss_type = loss_type
        if self.loss_type == 'binary'
            # define loss function as binary entropy loss for 17 classes
            loss_el = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels = self.y,
                            logits = self.logits,
                            name='elementwise_loss')

            # weight the lass
            if weights_classes is not None:
                # weight the loss of the classes.
                print('Weight the loss of each class with weights', weights_classes)
                loss_el = tf.multiply(loss_el, tf.convert_to_tensor(weights_classes, dtype=tf.float32))

            self.mean_loss = tf.divide(tf.reduce_sum(loss_el), tf.cast(tf.shape(loss_el)[0], tf.float32))
            self.loss_binary = self.mean_loss
            
        elif self.loss_type = 'binary_softmax':
            # define softmax loss for first 4 weather classes.
            loss_weather = tf.nn.softmax_cross_entropy_with_logits(
                            labels = self.y[:,:4],
                            logits = self.logits[:,:4],
                            name='softmax_loss')
            
            self.loss_softmax = tf.reduce_mean(loss_weather)
            # define loss function as binary entropy loss for 13 landform classes.
            loss_el = tf.nn.sigmoid_cross_entropy_with_logits(
                            labels = self.y[:,4:],
                            logits = self.logits[:,4:],
                            name='elementwise_loss')

            # weight the lass
            if weights_classes is not None:
                # weight the loss of the classes.
                print('Weight the loss of each class with weights', weights_classes[4:])
                loss_el = tf.multiply(loss_el, tf.convert_to_tensor(weights_classes[4:], dtype=tf.float32))
            
            self.loss_binary = tf.divide(tf.reduce_sum(loss_el), tf.cast(tf.shape(loss_el)[0], tf.float32))
            self.mean_loss = self.loss_softmax + self.loss_binary
        
        # define optimizer and set learning rate
        self.learning_rate = learning_rate
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            # Ensures that we execute the update_ops before performing the train_step
            #optimizer = tf.train.RMSPropOptimizer(learning_rate) # select optimizer and set learning rate
            optimizer = tf.train.AdamOptimizer(learning_rate)
            self.train_step = optimizer.minimize(self.mean_loss)

        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        else:
            raise ValueError('Must specify weights file path and sess')
        self.best_fs = -1
        self.best_model_path = None

    def run_model(self, session, Xd, yd, X_val = None, y_val = None,
                  epochs=1, batch_size=64, print_every = 100, training = False, 
                  plot_losses = False, verbose=False, checkpoint_path = None, 
                  data_augmentation=False):
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
        self.loss_binary = []
        self.loss_softmax = []
        self.train_fscore = []
        self.train_fscore_batch = []
        self.val_fscore = []
        
        if data_augmentation:
            datagen = helpers.batch_data_generator_train(Xd, yd, batch_size)
        
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
                if data_augmentation:
                    x_batch, y_batch = datagen.next()
                else:
                    x_batch = Xd[idx,:,:,:]
                    y_batch = yd[idx,:]
                
                # create a feed dictionary for this batch
                feed_dict = {self.x: x_batch,
                             self.y: y_batch,
                             self.lr: self.lr_var,
                             self.istraining: True}
            
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)
                fs_batch      = session.run(fs, feed_dict=feed_dict)
                self.train_fscore_batch.append(fs_batch)
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
#                        print('Elapse time from last print %f' % (toc_b-tic_b))
                        
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
                corr    = 0
                val_indicies = np.arange(X_val.shape[0])
                
                for i in range(int(math.ceil(X_val.shape[0]/batch_size))):
                    # generate indicies for the batch
                    start_idx = (i*batch_size)%X_val.shape[0]
                    idx = val_indicies[start_idx:start_idx+batch_size]

                    # create a feed dictionary for this batch
                    feed_dict_val = {self.x: X_val[idx,:],
                                     self.y: y_val[idx],
                                     self.lr: self.lr_var,
                                     self.istraining: False}
                    
#                    feed_dict_val_false = {self.x: X_val[idx,:],
#                                     self.y: y_val[idx],
#                                     self.lr: self.lr_var,
#                                     self.istraining: False}
                    # get batch size
                    actual_batch_size = y_val[i:i+batch_size].shape[0]
                    
                    acc_val_b, fs_val_b, corr_b = session.run([accuracy, fs, correct_prediction],feed_dict=feed_dict_val)
#                    acc_val_b_f, fs_val_b_f, corr_b_f = session.run([accuracy, fs, correct_prediction],
#                                                              feed_dict=feed_dict_val_false)
                    corr = corr + corr_b
#                    print('#####batch val acc and fs, corr_b, train is True:',acc_val_b, fs_val_b, corr_b)
#                    print('#####batch val acc and fs, corr_b, train is false:',acc_val_b_f, fs_val_b_f, corr_b_f)
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

    def predict_tag(self, session, X_test, threshold = 0.5, label_list = None, batch_size = 100):
            
        """This may not work when X_test is too big, consider breaking it into batches"""
        y_out       = logits2y(self.logits, threshold= threshold)
        y_test_feed = np.zeros((X_test.shape[0], self.num_classes))
        y_test      = np.zeros((X_test.shape[0], self.num_classes))
        
        test_indicies = np.arange(X_test.shape[0])
        for i in range(int(math.ceil(X_test.shape[0]/batch_size))):
            # generate indicies for the batch
            start_idx = (i*batch_size)%X_test.shape[0]
            idx = test_indicies[start_idx:start_idx+batch_size]

            # create a feed dictionary for this batch
            feed_dict_test = {self.x: X_test[idx,:],
                             self.y: y_test_feed[idx],
                             self.lr: self.lr_var,
                             self.istraining: False}

            y_pred_batch  = session.run(y_out,feed_dict=feed_dict_test)
            y_test[idx,:] = y_pred_batch

        tag_test   = None
        if label_list:
            tag_test   =  y2tags(y_test, label_list)
        return y_test, tag_test
        
    def val_acc(self, session, X_val, y_val, threshold=0.5):
        """evaluate accuracy"""
        y_out      =  logits2y(self.logits, threshold=threshold)
        acc_v, corrects_v  = correct_tags(self.y, y_out)
        feed_dict = {self.x: X_val,
                     self.y: y_val,
                     self.lr: 0,
                     self.istraining: False}
        variables = [acc_v, corrects_v]
        acc,  corrects =  session.run(variables, feed_dict=feed_dict)
        return acc, corrects

    def convlayers(self):
        """Define convolutional layers"""
        self.parameters = []
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            if not self.x_mean is None:
                self.x -= self.x_mean
            if not self.x_std is None:
                self.x /= self.x_std 
            
        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                             initializer = tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                             trainable = False)
            #
                           
            conv = tf.nn.conv2d(self.x, kernel, [1, 1, 1, 1], padding='SAME')
            
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
                              trainable = True)
                                
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = True)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]
            

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
        
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = True)
                                
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = True)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable('weights', dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = True)
                                
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = True)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope.name)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
        
        """Define fully connected layers"""
        
        with tf.variable_scope('fc') as scope:
            # flatten the output
            out = tf.contrib.layers.flatten(self.pool5)

            #FC 1
            out = fc_bn_relu(out, 4096, self.istraining)
            # add drop out
            out = tf.layers.dropout(out, rate = self.drop_out, training=self.istraining)

            #FC 2
            out = fc_bn_relu(out, 4096, self.istraining)
            # add drop out
            out = tf.layers.dropout(out, rate = self.drop_out, training=self.istraining)
            
            self.logits = tf.layers.dense(out, self.num_classes)
        

    def load_weights(self, weight_file, sess):
        """Assign pretrained weights and biases to all layers from weight_file"""
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        n = len(keys)
        
        #throw away the weights of all the fully connected layers.
        for i, k in enumerate(keys):
            if not 'fc' in k:
                sess.run(self.parameters[i].assign(weights[k]))
            else:
                print('%s is not assigned but initialized' % (k))
        
        vars_train = tf.trainable_variables()
        names = []
        for var in vars_train:
            names.append(var.name)
        print('trainable variables: ', names)
                
    
# some utility functions
def logits2y(logits, threshold=0.5):
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

def lr_decay(lr_max, epoch, decay_rate = 0.85, max_epoch = 5):
    
    if epoch < max_epoch:
        lr = lr_max
    else:
        lr = lr_max*decay_rate**(epoch-max_epoch)
    return lr
    