import numpy as np
import os
import scipy
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time
import skimage.io as io
import collections

def read_labels(csv_file):
    """
    function to read csv file and output filenames, tags, 
    unique labels and y hot vector for each train or test file.
    """
    image_names, tags = read_csv(csv_file)
    label_list = labels_unique()
    y = tags2onehot(tags, label_list)
    
    return y, label_list, image_names, tags
 
def read_csv(csv_file):
    """
    function to read csv file and output filenames, tags
    """
    image_names = []
    tags = []
    label_list = []

    with open(csv_file) as f:
        lines = f.readlines()[1:]
    
    for line in lines:
        if line:
            strs = line.strip().split(',')
            image_names.append(strs[0])
            tags.append(strs[1])
    return image_names, tags
    
def labels_unique():
    """
    mannually input the unique labels in the order of weather condition,
    common landforms and rare labels.
    """
    label_list = ['clear','haze', 'partly_cloudy','cloudy',
                  'primary', 'agriculture', 'water', 'habitation', 
                  'road', 'cultivation', 'blooming', 'selective_logging',
                  'slash_burn', 'conventional_mine', 'bare_ground', 
                  'artisinal_mine', 'blow_down']
    return label_list
    
def tags2onehot(tags, label_list):
    """
    using unique label_list to convert tags into a numpy array
    with each row a one hot vector if the label is present.
    """
    m = len(label_list)
    n = len(tags)
    y = np.zeros((n,m))
    
    for i in range(n):
        tags_i = tags[i].split()
        for tag in tags_i:
            index = label_list.index(tag)
            y[i, index] = 1
    return y
    
def load_image(path):
    """
    Return a 4D (r, g, b, nir) numpy array with the data in the specified TIFF filename.
    if the image is a .jpg, return 3D (r g b) numpy array
    """
    if os.path.exists(path):
        if '.tif' in path:
            A = io.imread(path)
            A = A[:,:,[2,1,0,3]]
        elif '.jpg' in path:
            A = plt.imread(path)[:,:,:3]
        return A
    else:
        raise ValueError('could not find image in %s in' % (path))

def check_paths(data_root):
    """check existence of path, if not existence, create."""
    train_csv_path = os.path.join(data_root, 'train.csv')
    test_csv_path = os.path.join(data_root, 'test.csv')
    train_jpg_path = os.path.join(data_root, 'train-jpg/')
    test_jpg_path = os.path.join(data_root, 'test-jpg/')
    test_jpg_path_a = os.path.join(data_root, 'test-jpg-additional/')
    train_tif_path = os.path.join(data_root,'train-tif')
    test_tif_path = os.path.join(data_root,'test-tif')
    
    paths = (train_csv_path, test_csv_path, train_jpg_path, 
             test_jpg_path, test_jpg_path_a, train_tif_path, test_tif_path)
    
    print('Necessary data:')
    for path in paths:
        check_path = os.path.exists(path)
        if check_path:
            print(path)
        else:
            print('Path %s doesn''t exist, a empty folder is created' % (path))
            os.makedirs(path)
    return paths
    
def process_raw_data_train(data_root, train_ratio = 0.9, save_processed_data=True, paths = None):
    """
    process raw training and test data from subfolders of data_root folder.
    train_ratio: the ratio of training data from original data.
    
    output (cast into 'uint8' data type):
    x_train, y_train
    x_val,   y_val
    """
    
    if paths:
        train_csv, test_csv, train_jpg_path, \
        test_jpg_path, test_jpg_path_a, train_tif_path, test_tif_path = paths
    else:
        train_csv, test_csv, train_jpg_path, \
            test_jpg_path, test_jpg_path_a, train_tif_path, test_tif_path= check_paths(data_root)
    
    # read in the filenames from training data.
    y, label_list, image_names, _ = read_labels(train_csv)
    y = y.astype('uint8')
    
    N = y.shape[0]
    # read in the image files 
    x = np.zeros((N, 256, 256, 4), dtype='uint8')
    count = 0
    print('Processing training data .....')
    tic = time.time()
    for img in image_names:
        img_path_jpg = os.path.join(train_jpg_path, img+'.jpg')
        img_path_tif = os.path.join(train_tif_path, img+'.tif')
        # read in the rgb
        x[count,:,:,0:3] = load_image(img_path_jpg).astype('uint8')
        # read in the nir channel from tif
        if os.path.exists(img_path_tif):
            x[count,:,:,-1] = (load_image(img_path_tif)[:,:,-1]//256).astype('uint8')
        count += 1
        if count%2000==0:
            toc = time.time()
            print('Processing %d-th image in total %d images, elapsed time %f' %(count, N, toc - tic))
    
    t_elapse = time.time() - tic
    print('Elapsed time: %f' % (t_elapse))
    print('Done!')
    
    # now take train_ratio of the training set as training training set
    # the rest is used for validation.
    
    N = y.shape[0]
    N_train = math.floor(N * train_ratio/1000)*1000
    N_val   = N - N_train
    
    mask_train = np.random.choice(N, size = N_train, replace= False)
    mask_val   = np.setdiff1d(np.arange(N), mask_train)
    
    x_train = x[mask_train,:,:,:]
    y_train = y[mask_train,:]
    x_val   = x[mask_val,:,:,:]
    y_val   = y[mask_val,:]
    
    if save_processed_data:
        print('Saving data_train_processed...')
        np.savez(os.path.join(data_root, 'data_train_processed'), 
                x_train, y_train, x_val, y_val)
        print('Done!')
    
    return x_train, y_train, x_val, y_val
    
def process_raw_data_test(data_root, save_processed_data=True):
    """
    process raw training and test data from subfolders of data_root folder.
    train_ratio: the ratio of training data from original data.
    
    output (cast into 'uint8' data type):
    x_train, y_train
    x_val,   y_val
    x_test,  y_test
    """
    train_csv, test_csv, train_jpg_path, \
    test_jpg_path, test_jpg_path_a, train_tif_path, test_tif_path= check_paths(data_root)
    
    # read in the filenames from training data.
    _, label_list, _, _ = read_labels(train_csv)
    y = y.astype('uint8')
   
    # read in the filenames from test data.
    y_test, _, image_names_test, _ = read_labels(test_csv)
    y_test = y_test.astype('uint8')
    
    N = y_test.shape[0]
    # read in the image files 
    x_test = np.zeros((N, 256, 256, 4), dtype='uint8')
    
    count = 0
    print('Processing testing data .....')
    tic = time.time()
    for img in image_names_test:
        img_path_jpg = os.path.join(test_jpg_path, img+'.jpg')
        img_path_jpg_a = os.path.join(test_jpg_path_a,img+'.jpg')
        img_path_tif = os.path.join(test_tif_path, img+'.tif')
        
        if os.path.exists(img_path_jpg_a):
            img_path_jpg = img_path_jpg_a

        # read in the rgb
        x_test[count,:,:,0:3] = load_image(img_path_jpg).astype('uint8')
        # read in the nir channel from tif
        # if tif doesnt't exist, the channel nir will be 0
        if os.path.exists(img_path_tif):
            x_test[count,:,:,-1] = (load_image(img_path_tif)[:,:,-1]//256).astype('uint8')
        count +=1
        if count%2000==0:
            toc = time.time()
            print('Processing %d-th image in total %d images, elapsed time %f' %(count, N, toc - tic))
    t_elapse = time.time() - tic
    print('Elapsed time: %f' % (t_elapse))
    print('Done!')
    
    if save_processed_data:
        print('Saving data_test_processed...')
        np.savez(os.path.join(data_root, 'data_test_processed'), 
                x_test, y_test)
        print('Done!')
    return x_test, y_test

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


def multi_f_beta_score_np(y, y_out, beta=2.0):
    """compute mean F_beta score"""
    eps = 1e-6
    y = y.astype(np.float)
    y_out = y_out.astype(np.float)
    num_pos = np.sum(y_out, axis=1)
    tp      = np.sum(y_out*y,axis=1)
    num_pos_hat = np.sum(y, axis=1)
    precision = tp/(num_pos+eps)
    recall    = tp/(num_pos_hat+eps)
    fs = (1+beta*beta)*precision*recall/(beta*beta*precision+recall+eps)
    f  = np.mean(fs,axis=0)
    return f


def balance_data(X, y, label_list, top = 5):
    """Downsample the data with the most frequent tags to balance the dataset
    Args:
    X: training data, [num_train, height, width, channels]
    y: training labels, [num_train, num_class]
    label_list: a list of unique labels.
    top: number of most frequenct tags to be downsampled.
    
    Return:
    X_out: [new_num_train, height, width, channels]
    y_out: [new_num_train, num_class]
    """
    tags = y2tags(y, label_list)
    n = len(tags)
    c = collections.Counter(tags)
    tops = c.most_common(top)
    top_tags, top_counts = zip(*tops)
    print('20 most common tags:')
    print(c.most_common(20))
    num_cut = top_counts[-1]
    
    # create booleen array for data not in the top list
    b_top = []
    b_none_top = []
    for tag in tags:
        if tag in top_tags:
            b_top.append(True)
            b_none_top.append(False)
        else:
            b_top.append(False)
            b_none_top.append(True)
    b_top = np.array(b_top, dtype=np.bool)
    b_none_top = np.array(b_none_top, dtype=np.bool)
    
    X_out = X[b_none_top,:,:,:]
    y_out = y[b_none_top,:]
    
    print('none top data number: %d' % (X_out.shape[0]))
    
    # balance the data in the top list.
    tags_array = np.array(tags,dtype=np.string_)
    for tag, count in tops:
        b = []
        for tagi in tags:
            if tagi==tag:
                b.append(True)
            else:
                b.append(False)
        b = np.array(b,dtype=np.bool) 
                
        X_tag = X[b,:,:,:]
        y_tag = y[b,:]
        mask  = np.random.choice(count, num_cut, replace = False)
        X_out = np.concatenate((X_out, X_tag[mask,:,:,:]), axis = 0)
        y_out = np.concatenate((y_out, y_tag[mask,:]), axis = 0)
    
    # random permute X and y
    idx = np.random.permutation(X_out.shape[0])
    X_out = X_out[idx,:,:,:]
    y_out = y_out[idx,:]
    
    print('Total number of data: %d' % X_out.shape[0])
    tags = y2tags(y_out, label_list) 
    c = collections.Counter(tags)
    print('20 most common tags after balancing the dataset:')
    print(c.most_common(20))
    
    return X_out, y_out
    
def deleteTags(X, y, tags_del = [0,1,2,3,4], label_list = None, retain = 0.1):
    """remove the weather tags and primary tag from the data.
    Args:
    X: training data, [N, height, width, channels]
    y: training labels, [N, num_class]
    tags_del: a list of tag index to be deleted
    label_list: a list of unique labels.
    retrain: the fraction of retained samples with labels all zero.
    
    Return:
    X_out: [new_num_train, height, width, channels]
    y_out: [new_num_train, num_class]
    """
    
    N, num_class = y.shape
    mask = np.ones(num_class, dtype=bool)
    mask[tags_del] = False
    
    y_out = y[:, mask]
    mask_keep = np.sum(y_out,axis=1) > 0
    mask_del = np.sum(y_out,axis=1) == 0
    
    X_del = X[mask_del]
    y_del = y_out[mask_del]

    y_out = y_out[mask_keep]
    X_out = X[mask_keep]
    
    n = X_out.shape[0]
    
    retain_num = np.round(X_del.shape[0]*retain).astype(int)
    retain_idx = np.random.choice(X_del.shape[0], retain_num, replace=False)
    
    X_out = np.concatenate((X_out, X_del[retain_idx]), axis=0)
    y_out = np.concatenate((y_out, y_del[retain_idx,:]), axis=0)
    
    # random permute
    idx = np.random.permutation(y_out.shape[0])
    X_out = X_out[idx]
    y_out = y_out[idx]
    
    label_list_new = []
    if label_list:
        label_list_new = [i for j, i in enumerate(label_list) if j not in tags_del]
        
    return X_out, y_out, label_list_new

def mean_std_batch(x, batch_size = 2000):
    n          = x.shape[0]
    indicies   = np.arange(n)
    sums = np.zeros_like(x[0,...],dtype=np.float32)
    mean = np.zeros_like(sums)
    var = np.zeros_like(sums)
    std = np.zeros_like(sums)
    tic = time.time()
    # compute the mean
    for i in range(int(math.ceil(n/batch_size))):
        start_idx = (i*batch_size)%n
        idx = indicies[start_idx:start_idx+batch_size]
        sums += np.sum(x[idx,...].astype(np.float32),axis=0)
    mean = sums/n
    
    print('done with mean', mean.shape)
    
    for i in range(int(math.ceil(n/batch_size))):
        start_idx = (i*batch_size)%n
        idx = indicies[start_idx:start_idx+batch_size]
        var += np.sum((x[idx,...].astype(np.float32) - mean)
                      *(x[idx,...].astype(np.float32) - mean),axis=0)
    var = var/n
    std = np.sqrt(var)
    print('done with std', std.shape)
    return mean, std
    