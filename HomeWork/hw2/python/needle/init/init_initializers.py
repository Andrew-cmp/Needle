import math
from .init_basic import *

####一开始使用kwargs['shape']来传递shape参数，但其实rand的第一个参数时*arge，所以需要传递的其实是元组或者*[]，并且shape也找错了，应该是fan_in和fan_out
def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    assert gain > 0, "Gain must be positive"
    a = gain*math.sqrt(6/(fan_in + fan_out))
    #return rand(kwargs['shape'], low=-1*a, high=a)
    return rand(*[fan_in,fan_out], low=-1*a, high=a,**kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    assert gain > 0, "Gain must be positive"
    a = gain*math.sqrt(2/(fan_in + fan_out))
    a_squared = a**2
    #return randn(kwargs['shape'], mean=0, std=a_squared)
    return randn(*[fan_in,fan_out], mean=0, std=a,**kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    a = math.sqrt(6/fan_in)
    #return rand(kwargs['shape'], low=-1*a, high=a)
    return rand(*[fan_in,fan_out], low=-1*a, high=a,**kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    a = math.sqrt(2/fan_in)
    a_square = 2/fan_in
    #return randn(kwargs['shape'], mean=0, std=a_square)
    return randn(*[fan_in,fan_out], mean=0, std=a,**kwargs)
    ### END YOUR SOLUTION
