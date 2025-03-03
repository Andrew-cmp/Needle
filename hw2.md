# hw2

## question1

在第一个问题中，您将实现几种不同的权重初始化方法。这将在 `python/needle/init/init_initializers.py` 文件中完成，该文件包含许多使用各种随机和恒定初始化来初始化针张量的例程。按照与现有初始化程序相同的方法（您将希望从下面的函数中调用例如在python/pine/init/init_basic.py中实现的init.rand或init.randn），实现以下常见的初始化方法。在所有情况下，函数都应该通过fan_out 2D张量返回fan_in（可以通过例如整形来扩展到其他大小）。

![image-20240302190621742](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302190621742.png)

这些function其实就是用来对tensor进行初始化的函数，只不过初始化的分布和参数有要求。

### Xavier uniform

`xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs)`

![image-20240302185959352](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302185959352.png)

### Xavier normal

`xavier_normal(fan_in, fan_out, gain=1.0, **kwargs)`

![image-20240302190958567](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302190958567.png)

### Kaiming uniform

`kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs)`

![image-20240302191502249](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302191502249.png)

### Kaiming normal

`kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs)`

![image-20240302191811230](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302191811230.png)

### 反思

一开始使用kwargs['shape']来传递shape参数，但其实rand的第一个参数时*arge，所以需要传递的其实是元组或者*[]，并且shape也找错了，应该是fan_in和fan_out

```python
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    assert gain > 0, "Gain must be positive"
    a = gain*math.sqrt(6/(fan_in + fan_out))
    #return rand(kwargs['shape'], low=-1*a, high=a)
    return rand(*[fan_in,fan_out], low=-1*a, high=a)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    assert gain > 0, "Gain must be positive"
    a = gain*math.sqrt(2/(fan_in + fan_out))
    a_squared = a**2
    #return randn(kwargs['shape'], mean=0, std=a_squared)
    return randn(*[fan_in,fan_out], mean=0, std=a)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    a = math.sqrt(6/fan_in)
    #return rand(kwargs['shape'], low=-1*a, high=a)
    return rand(*[fan_in,fan_out], low=-1*a, high=a)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    a = math.sqrt(2/fan_in)
    a_square = 2/fan_in
    #return randn(kwargs['shape'], mean=0, std=a_square)
    return randn(*[fan_in,fan_out], mean=0, std=a)
    ### END YOUR SOLUTION
```

# question2

这一节需要实现一些module，路径在 `python/needle/nn/nn_basic.py`中，具体来说，对于下面描述的以下module，初始化constructor函数中模块的任何变量，并填写forward function。注意：请确保您使用的是Question1中初始化参数的init函数，不要忘记传递dtype参数。

![image-20240302221245851](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302221245851.png)