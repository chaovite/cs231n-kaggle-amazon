import numpy as np
import os
import scipy
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import time
import skimage.io as io

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
