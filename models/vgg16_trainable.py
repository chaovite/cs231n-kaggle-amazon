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
from scipy.misc import imread, imresize

# some utility functions
def logits2y(logits, threshold):
    """convert logitst to predictions"""
    return tf.cast(tf.less(threshold, self.logits), tf.int32)

def correct_tags(y, yout):
    """compare y and yout and output the accuracy"""
    y = tf.cast(y, tf.int32)
    n = tf.shape(y)[0]
    diff    = tf.reduce_max(tf.abs(tf.subtract(y, y_out)), 1)
    wrongs  = tr.reduce_sum(diff)
    correct = tf.cast(n - wrongs, tf.float32)
    acc     = tf.cast(correct/n, tf.float32)
    return correct, acc
    
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
    
class vgg16_trainable:
    """ A trainable vgg16 model that allow fine tuning the last fc layer."""

    def __init__(self, weights, sess, height=224, width=224, channel=3, x_mean = None):
        
        self.x = tf.placeholder(tf.float32, [None, height, width, channel])
        self.y = tf.placeholder(tf.float32, [None, num_classes])
        self.x_mean = x_mean
        
        # build neural network architecture.
        self.convlayers()
        self.fc_layers()
        if weights is not None and sess is not None:
            self.load_weights(weights, sess)
        else:
            raise ValueError('Must specify weights file path and sess')
        self.best_acc = -1
        self.best_model_path = None

    def convlayers(self):
    """Define convolutional layers"""
        self.parameters = []
        # zero-mean input
        with tf.name_scope('preprocess') as scope:
            if not self.x_mean
                self.x_mean = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32, shape=[1, 1, 1, 3], name='x_mean')
            images = self.x - mean

        # conv1_1
        with tf.variable_scope('conv1_1') as scope:
            kernel = tf.get_variable('weights', shape=[3, 3, 3, 64], dtype= tf.float32,
                             initializer = tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1),
                             trainable = False)
                            
            conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[64], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv1_2
        with tf.variable_scope('conv1_2') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 3, 64], dtype= tf.float32,
                             initializer = tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1),
                             trainable = False)
                             
            conv = tf.nn.conv2d(self.conv1_1, kernel, [1, 1, 1, 1], padding='SAME')
            biases = tf.get_variable('biases', shape=[64], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv1_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool1
        self.pool1 = tf.nn.max_pool(self.conv1_2,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool1')

        # conv2_1
        with tf.variable_scope('conv2_1') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 64, 128], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.pool1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[128], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv2_2
        with tf.variable_scope('conv2_2') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 128, 128], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv2_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[128], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv2_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool2
        self.pool2 = tf.nn.max_pool(self.conv2_2,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool2')

        # conv3_1
        with tf.name_scope('conv3_1') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 128, 256], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.pool2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[256], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_2
        with tf.variable_scope('conv3_2') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 256, 256], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv3_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[256], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv3_3
        with tf.variable_scope('conv3_3') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 256, 256], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv3_2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[256], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv3_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool3
        self.pool3 = tf.nn.max_pool(self.conv3_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool3')

        # conv4_1
        with tf.variable_scope('conv4_1') as scope:
        
            kernel = tf.get_variable('weights',shape=[3, 3, 256, 512], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.pool3, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[512], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            
        # conv4_2
        with tf.variable_scope('conv4_2') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 512, 512], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv4_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[512], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv4_3
        with tf.variable_scope('conv4_3') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 512, 512], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv4_2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[512], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv4_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool4
        self.pool4 = tf.nn.max_pool(self.conv4_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

        # conv5_1
        with tf.variable_scope('conv5_1') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 512, 512], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.pool4, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[512], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_1 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]
            

        # conv5_2
        with tf.variable_scope('conv5_2') as scope:
        
            kernel = tf.get_variable('weights',shape=[3, 3, 512, 512], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv5_1, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[512], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_2 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # conv5_3
        with tf.variable_scope('conv5_3') as scope:
            kernel = tf.get_variable('weights',shape=[3, 3, 512, 512], dtype= tf.float32,
                              initializer =tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1),
                              trainable = False)
                                
            conv = tf.nn.conv2d(self.conv5_2, kernel, [1, 1, 1, 1], padding='SAME')
            
            biases = tf.get_variable('biases', shape=[512], dtype= tf.float32,
                             initializer = tf.constant(0.0, shape=[512], dtype=tf.float32),
                             trainable = False)
            out = tf.nn.bias_add(conv, biases)
            self.conv5_3 = tf.nn.relu(out, name=scope)
            self.parameters += [kernel, biases]

        # pool5
        self.pool5 = tf.nn.max_pool(self.conv5_3,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME',
                               name='pool4')

    def fc_layers(self):
    """Define fully connected layers"""
        # fc1
        with tf.variable_scope('fc1') as scope:
            shape = int(np.prod(self.pool5.get_shape()[1:]))
            
            fc1w = tf.get_variable('weights', shape =[shape, 4096], dtype= tf.float32,
                                initializer = tf.truncated_normal([shape, 4096], dtype=tf.float32, stddev=1e-1),
                                trainable = False)
            fc1b = tf.get_variable('biases', shape =[4096], dtype= tf.float32,
                                initializer = tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                trainable = False)
            pool5_flat = tf.reshape(self.pool5, [-1, shape])
            fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
            self.fc1 = tf.nn.relu(fc1l)
            self.parameters += [fc1w, fc1b]

        # fc2
        with tf.variable_scope('fc2') as scope:
        
            fc2w = tf.get_variable('weights', shape =[4096, 4096], dtype= tf.float32,
                                initializer = tf.truncated_normal([4096, 4096], dtype=tf.float32, stddev=1e-1),
                                trainable = False)
            fc2b = tf.get_variable('biases', shape =[4096], dtype= tf.float32,
                                initializer = tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                trainable = False)             
            fc2l = tf.nn.bias_add(tf.matmul(self.fc1, fc2w), fc2b)
            
            self.fc2 = tf.nn.relu(fc2l)
            self.parameters += [fc2w, fc2b]

        # fc3 # freeze the weights and bias of all the other layers and only train this fully connected layer.
        with tf.variable_scope('fc3') as scope:
        
            fc3w = tf.get_variable('weights', shape =[4096, 17], dtype= tf.float32,
                                initializer = tf.truncated_normal([4096, 17], dtype=tf.float32, stddev=1e-1),
                                trainable = True)
            fc3b = tf.get_variable('biases', shape =[17], dtype= tf.float32,
                                initializer = tf.constant(1.0, shape=[4096], dtype=tf.float32),
                                trainable = True) 
                                 
            self.fc3l = tf.nn.bias_add(tf.matmul(self.fc2, fc3w), fc3b)
            self.logits = fc3l
            self.parameters += [fc3w, fc3b]

    def load_weights(self, weight_file, sess):
        """Assign pretrained weights and biases to all layers from weight_file"""
        weights = np.load(weight_file)
        keys = sorted(weights.keys())
        for i, k in enumerate(keys):
            print i, k, np.shape(weights[k])
            sess.run(self.parameters[i].assign(weights[k]))
    
    def run_model(self, session, Xd, yd, X_val = None, y_val = None,
                  epochs=1, batch_size=64, print_every = 100, learning_rate=5e-4,
                  training = None, plot_losses = False, verbose=False, checkpoint_path = None):
          """method for training and evaluation """

        # define our optimizer
        optimizer  = tf.train.AdamOptimizer(learning_rate) # select optimizer and set learning rate
        train_step = optimizer.minimize(self.mean_loss)
        
        # convert logits to label prediction.
        y_out      =  logits2y(self.logits, 0.5)
        
        accuracy, correct_prediction  = correct_tags(self.y, y_out)
        
        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None
    
        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [mean_loss,correct_prediction,accuracy]
        
        if training_now:
            # if it's training, replace accuracy with training step.
            variables[-1] = training
    
        # counter 
        iter_cnt = 0
        validation_acc= []
        train_acc = []
        losses = []
        
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                # generate indicies for the batch
                start_idx = (i*batch_size)%X_train.shape[0]
                idx = train_indicies[start_idx:start_idx+batch_size]
            
                # create a feed dictionary for this batch
                feed_dict = {self.x: tf.convert_to_tensor(Xd[idx,:], dtype = tf.float32),
                             self.y: tf.convert_to_tensor(yd[idx], dtype = tf.int32)}
                # get batch size
                actual_batch_size = yd[i:i+batch_size].shape[0]
            
                # have tensorflow compute loss and correct predictions
                # and (if given) perform a training step
                loss, corr, _ = session.run(variables,feed_dict=feed_dict)
            
                # aggregate performance stats
                losses.append(loss*actual_batch_size)
                correct += np.sum(corr)
            
                # print every now and then
                if verbose:
                    if training_now and (iter_cnt % print_every) == 0:
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
                              .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
                iter_cnt += 1
            total_correct = correct/Xd.shape[0]
            total_loss = np.sum(losses)/Xd.shape[0]

            print("Epoch {2}, Overall loss = {0:.3g} and training accuracy of {1:.3g}"\
                  .format(total_loss,total_correct,e+1))
        
            # evaluate the evaluation accuracy.
            is_val = not ((X_val is None) or (y_val is None))
            train_acc.append(total_correct)
            
            saver = tf.train.Saver()
            if is_val:            
                # create a feed dictionary for evaluation.
                feed_dict_val = {self.x: X_val,
                                 self.y: y_val}
                acc_val = session.run(accuracy,feed_dict=feed_dict_val)
                print("Epoch %d, validation accuracy of %f"\
                  % (e+1, acc_val))
                validation_acc.append(acc_val)
                
                if acc_val > self.best_acc:
                    self.best_acc = acc_val
                    if checkpoint_path:
                        save_path = saver.save(sess, checkpoint_path)
                        self.best_model_path = save_path
                        print('Best pretrained vgg model saved in %s' % (save_path))

            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e+1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()

        return total_loss, total_correct, losses, validation_acc, train_acc

        def predict_tag(self, session, X_test, label_list = None):
            
            """This may not work when X_test is too big, consider breaking it into batches"""
            y_out      =  logits2y(self.logits, 0.5)
            feed_dict_predict = {self.x: X_test,
                                 self.y: None}
            y_test     =  session.run(y_out, feed_dict=feed_dict_predict)
            tag_test   = None
            if label_list:
                tag_test   =  y2tags(y_test, label_list)
            return y_test, tag_test
        
        def val_acc(self, session, X_val, y_val):
            """evaluate accuracy"""
            y_out      =  logits2y(self.logits, 0.5)
            acc_v, corrects_v  = correct_tags(self.y, y_out)
            feed_dict = {self.x: X_val,
                                 self.y: y_val}
            variables = [acc_v, corrects_v]
            acc,  corrects =  session.run(variables, feed_dict=feed_dict)
            return acc, corrects

if __name__ == '__main__':
    sess = tf.Session()
    imgs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg = vgg16(imgs, 'vgg16_weights.npz', sess)

    img1 = imread('laska.png', mode='RGB')
    img1 = imresize(img1, (224, 224))

    prob = sess.run(vgg.probs, feed_dict={vgg.imgs: [img1]})[0]
    preds = (np.argsort(prob)[::-1])[0:5]