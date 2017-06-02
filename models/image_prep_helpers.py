import tensorflow as tf
import numpy as np

def preprocess_for_train_single_image(image, output_height, output_width):
    """Preprocesses a single image for training.
    Args:
    image: A `Tensor` [height, width, channels]
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    Returns:
    A preprocessed images tensor [batch_size, output_height, output_width, channels]
    """
    rank_assertion = tf.Assert(
    tf.equal(tf.rank(image), 3),
    ['Rank of image must be equal to 3.'])
    assert(output_height==output_width, 'output_height and output_width must be equal')
    H, W, C = image.get_shape()
    
    # central crop
    central_fraction = float(output_height)/float(H)
    image = tf.image.central_crop(image, central_fraction)
    # random flip up and down, right and left.
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)
    #randomly transpose the image
    if np.random.rand(1) > 0.5:
        image = tf.image.transpose_image(image)
    
    #rotate images by random number of 90 degree.
    k = np.random.randint(4, size=1)
    if k!=0:
        image = tf.image.rot90(image, k = k)
    
    #adjust random brightness a little bit.
    max_delta = 0.2
    image = tf.image.random_brightness(image, max_delta)
    
    # randomly perturbing the saturation.
    saturation_lower = 0.9
    saturation_upper = 1.1
    image = tf.image.random_saturation(image, lower, upper)
    return image


def preprocess_for_eval_single_image(image, output_height, output_width):
    """Preprocesses a single image for evaluation.
    Args:
    image: A `Tensor` [height, width, channels]
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    Returns:
    A preprocessed images tensor [batch_size, output_height, output_width, channels]
    """
    rank_assertion = tf.Assert(
    tf.equal(tf.rank(image), 3),
    ['Rank of image must be equal to 3.'])
    assert(output_height==output_width, 'output_height and output_width must be equal')
    H, W, C = image.get_shape()
    
    # central crop
    central_fraction = float(output_height)/float(H)
    image = tf.image.central_crop(image, central_fraction)
    
    return image


def preprocess_for_train(images, output_height, output_width):
    
    """Preprocesses images for training.
    Args:
    image: A `Tensor` [batch_size, height, width, channels]
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    Returns:
    A preprocessed images tensor [batch_size, output_height, output_width, channels]
    """
    N, _ ,_ , channels = images.get_shape()
    # central crop
    images_processed = tf.zeros((N,output_height, output_width, channels))
    for i in range(N):
        images_processed[i,:,:,:] = \
        preprocess_for_train_single_image(images[i,:,:,:], output_height, output_width)
    return images_processed

def preprocess_for_eval(images, output_height, output_width):
    
    """Preprocesses images for evaluation.
    Args:
    image: A `Tensor` [batch_size, height, width, channels]
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    Returns:
    A preprocessed images tensor [batch_size, output_height, output_width, channels]
    """
    N, _ ,_ , channels = images.get_shape()
    # central crop
    images_processed = tf.zeros((N,output_height, output_width, channels))
    for i in range(N):
        images_processed[i,:,:,:] = \
        preprocess_for_eval_single_image(images[i,:,:,:], output_height, output_width)
    return images_processed
    

def preprocess_images(images, output_height, output_width, is_training=False):
    """Preprocesses the given image.
    Args:
    image: A `Tensor` [batch_size, height, width, channels]
    output_height: The height of the image after preprocessing.
    output_width: The width of the image after preprocessing.
    is_training: `True` if we're preprocessing the image for training and
      `False` otherwise.
    Returns:
    A preprocessed images tensor [batch_size, output_height, output_width, channels]
    """
    if is_training:
        return preprocess_for_train(images, output_height, output_width)
    else:
        return preprocess_for_eval(image, output_height, output_width)

