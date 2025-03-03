"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    ### https://xinancsd.github.io/MachineLearning/mnist_parser.html
    ImagePath = image_filename
    labelPath = label_filename
    print(image_filename)
    ##奶奶的，给的是压缩文件，得用gzip打开
    ImageData = gzip.open(ImagePath,'rb').read()
    labelData = gzip.open(labelPath,'rb').read()
    # read image
    offset = 0
    fmt_header ='>IIII'   # 以大端的方法读取4个unsight int32
    magic_num,num_image,num_rows,num_cols = struct.unpack_from(fmt_header,ImageData,offset)
 
    #print('魔数：{}，图片数：{}，row：{}'.format(magic_num,num_image,num_rows))
    offset += struct.calcsize(fmt_header)
    fmt_image = '>'+str(num_cols*num_rows)+'B'
    image = np.empty((num_image,num_cols*num_rows),np.float32)
    for i in range(num_image):
        im = struct.unpack_from(fmt_image,ImageData,offset)
        image[i] = np.array(im,np.float32)
        offset += struct.calcsize(fmt_image)

    #normalize https://datagy.io/python-numpy-normalize/
    min_val = np.min(image)
    max_val = np.max(image)
    normalize_image = (image-min_val) /(max_val-min_val)


    # read label
    offset = 0
    fmt_header ='>II'   # 以大端的方法读取4个unsight int32
    magic_num,num_label = struct.unpack_from(fmt_header,labelData,offset)
    offset += struct.calcsize(fmt_header)
    fmt_label = '>B'
    label = np.empty((num_label),np.uint8)
    for i in range(num_label):
        lb = struct.unpack_from(fmt_label,labelData,offset)
        label[i] = lb[0]
        offset += struct.calcsize(fmt_label)
     
    return (normalize_image, label)
    ### END YOUR CODE


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    power = ndl.exp(Z)
    sum = ndl.summation(power, axes=1)
    log = ndl.log(sum)
    loss = ndl.summation(log - Z * y_one_hot, axes=1)
    return ndl.mean(loss)
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W1
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]
    itera_num = int(np.floor(num_examples/batch))
    y_one_hot = np.zeros((num_examples,y.shape[0]))
    y_one_hot[np.arange(y.size), y] = 1
    for i in range(itera_num):
        X_batch = X[i*batch:i*batch+batch,:]
        y_one_hot_batch = y_one_hot[i*batch:i*batch+batch,:]
        loss = nn_batch(X_batch,y_one_hot_batch,W1,W2,lr)
        loss.backward()
        W1 = W1.realize_cached_data() - lr * W1.grad.realize_cached_data()
        W2 = W2.realize_cached_data() - lr * W2.grad.realize_cached_data()
    if(itera_num*batch<num_examples):
        X_batch = X[itera_num*batch:num_examples,:]
        y_one_hot_batch = y_one_hot[itera_num*batch:num_examples,:]
        loss = nn_batch(X_batch,y_one_hot_batch,W1,W2,lr)
        loss.backward()
        W1 = W1.realize_cached_data() - lr * W1.grad.realize_cached_data()
        W2 = W2.realize_cached_data() - lr * W2.grad.realize_cached_data()
    return W2
def nn_batch(X, y, W1, W2, lr):

    Z = ndl.relu(X.matmul(W1)).matmul(W2)
    return softmax_loss(Z, y)

    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
