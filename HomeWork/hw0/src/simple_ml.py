import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass


def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    a = x+y
    ### BEGIN YOUR CODE
    return a
    ### END YOUR CODE


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


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    #（在这里以一个样本为例）操，以前一直想错了，想成np.log(np.exp(Z).sum(axis=1)) -y了。y是lable，减的应该是这个label的概率Z[y]
    #Z = [1.2,2.2,3.2] y=1
    #softmax=exp(h1(x))/Σexp(hj(x)) = exp(1.2)/(exp(1.2)+exp(2.2)+exp(3.2))

    ### BEGIN YOUR CODE
    softmax_loss = [np.log(sum(np.exp(sample)))-sample[label]  for sample, label in zip(Z, y)]
    return np.mean(softmax_loss)
    # a =  np.log(np.exp(Z).sum(axis=1)) 
    # b =  np.take_along_axis(Z, np.expand_dims(y, axis=1), axis=1).squeeze()
    return np.mean(a - y)

    
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
        ### BEGIN YOUR CODE
    num_examples = X.shape[0]
    num_classes = theta.shape[1]
    num_feature = theta.shape[0]
    itera_num = int(np.floor(num_examples/batch))
    for i in range(itera_num):
        softmax_regression_batch(X[i*batch:i*batch+batch,:],y[i*batch:i*batch+batch],theta,lr)
    
    if(itera_num*batch<num_examples):
        softmax_regression_batch(X[itera_num*batch:num_examples,:],y[itera_num*batch:num_examples],theta,lr)




def softmax_regression_batch(X,y,theta,lr):

    h = np.exp(np.matmul(X, theta))
    Z = h/np.sum(h,axis = 1,keepdims=True)
    
    yi = np.expand_dims(y,axis=1)
    one_hot = np.zeros(Z.shape, dtype=np.int8)
    np.put_along_axis(one_hot,yi,1,axis=1)

    t = Z-one_hot
    deta = (np.matmul(X.T,t))/X.shape[0]

    np.subtract(theta, lr*deta, out=theta)


    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
#     num_examples = X.shape[0]
#     num_classes = W2.shape[1]
#     num_feature = X.shape[1]
#     itera_num = int(np.floor(num_examples/batch))
#     Z1 = RELU(X@W1)
#     Z2 = Z1@W2
#     for i in range(itera_num):
#         G2 = softmax_regression_batch(Z1[itera_num*batch:num_examples,:],y[itera_num*batch:num_examples],W2,lr)
#         G1 = RELU_regression_batch(Z2[i*batch:i*batch+batch,:],[i*batch:i*batch+batch],  W1,lr)
    
#     if(itera_num*batch<num_examples):
#         G2 = softmax_regression_batch(X[itera_num*batch:num_examples,:],y[itera_num*batch:num_examples],W2,lr)
#         G1 = RELU_regression_batch(X[itera_num*batch:num_examples,:],[itera_num*batch:num_examples],W1,lr)

# def RELU(X):
#     f = lambda i:  0 if i<0 else i 
#     return f(X)

# def RELU_regression_batch(X, y,theta lr):
#     h = np.exp(np.matmul(X, theta))
#     Z = h/np.sum(h,axis = 1,keepdims=True)
    
#     yi = np.expand_dims(y,axis=1)
#     one_hot = np.zeros(Z.shape, dtype=np.int8)
#     np.put_along_axis(one_hot,yi,1,axis=1)
    
    num_examples = X.shape[0]
    itera_num = int(np.floor(num_examples/batch))
    for i in range(itera_num):
        nn_batch(X[i*batch:i*batch+batch,:],y[i*batch:i*batch+batch],W1,W2,lr)
    
    if(itera_num*batch<num_examples):
        nn_batch(X[itera_num*batch:num_examples,:],y[itera_num*batch:num_examples],W1,W2,lr)



def nn_batch(X, y, W1, W2, lr):

    Z1 = X@W1
    Z1[Z1 <= 0] = 0
    #Z1 = np.where(t<=0,0,t)
    Z2 = Z1@W2



    h = np.exp(Z2)
    t = h/np.sum(h,axis = 1,keepdims=True)

    yi = np.expand_dims(y,axis=1)
    one_hot = np.zeros(t.shape, dtype=np.uint8)
    np.put_along_axis(one_hot,yi,1,axis=1)
    G2 = t-one_hot


    #f = lambda i:  0 if i<=0 else 1
    t = np.where(Z1<=0,0,1)
    G1 = t*(G2@W2.T)

    detaW2 = (Z1.T@G2)/X.shape[0]
    detaW1 = (X.T@G1)/X.shape[0]

    np.subtract(W2, lr*detaW2, out=W2)
    np.subtract(W1, lr*detaW1, out=W1)
    
    ### END YOUR CODE







### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
