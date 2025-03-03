# hw0

首先这是一个必须在google drive中写的作业，通过google drive中的colab软件打开hw0.ipnb文件查看作业要求，使用本地的vscode编写代码文件。

## Python加载mnist数据库

## softmax loss的计算方法

## softmax regression的随机梯度下降

题目描述

![image-20231129210223315](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20231129210223315.png)

要点

+ src中的代码需要编写的是`softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100)`函数，epoch是指所有的样本都要使用一边，并且每个bath的bath size给定。
+ 这里对z进行normalize，是直接对z的axis=1进行求和，不懂这种正则化是什么意思。
+ 这个地方求Iy的求法很重要学一学。
+ 要求your function should modify the `Theta`array in-place，由于python是对象传递，直接用theta=theta-lr*deta会直接创建一个新的theta而不会改变原有的值，所以我们选择使用np.subtract(theta, sub, out=theta)函数。

```python
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
```

## Question5 对两层的神经网络做SGD

![image-20231129222927792](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20231129222927792.png)

# hw1

