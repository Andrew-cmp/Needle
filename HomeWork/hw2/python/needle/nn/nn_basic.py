"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features,device=device,dtype=dtype,requires_grad=True))
        self.bias = Parameter(init.kaiming_uniform(out_features,1,device=device,dtype=dtype,requires_grad=True).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = X.matmul(self.weight) 
        if self.bias is not None:
            out += self.bias.broadcast_to(out.shape)
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X,(X.shape[0],-1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        x = ops.relu(x)
        return x
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = x
        for m in self.modules:
            out = m(out)
        return out
        
        
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        #return ops.summation(ops.log(ops.summation(ops.exp(logits),axes=(0,)) -init.one_hot(logits.shape[-1],y)*logits))
        # logits:m*n m是批量大小，n是类别数
        # y:m*1 m是批量大小
        one_hot_y = init.one_hot(logits.shape[1], y)
        #return ops.summation(ops.logsumexp(logits,axes=(1,)) - ops.summation(one_hot_y*logits,axes=(0,))
        # 第一个问题：没有考虑批量大小的平均，样本的loss应该是batch loss总和的平均
        return (ops.summation(ops.logsumexp(logits, (1,)) / logits.shape[0]) - ops.summation(one_hot_y * logits / logits.shape[0]))
        ### END YOUR SOLUTION


# class BatchNorm1d(Module):
#     def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
#         super().__init__()
#         self.dim = dim
#         self.eps = eps
#         self.momentum = momentum
#         ### BEGIN YOUR SOLUTION
#         self.weight =Parameter(init.ones(dim,device = device,dtype = dtype))
#         self.bias = Parameter(init.zeros(dim,device = device,dtype = dtype))
#         self.running_mean = init.zeros(dim,device = device,dtype = dtype)
#         self.running_var = init.ones(dim,device = device,dtype = dtype)
#         ### END YOUR SOLUTION

#     def forward(self, x: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         if(self.training):
            
#             mean = (x.sum((0,))/x.shape[0]).reshape((1,x.shape[1])).broadcast_to(x.shape)
#             var = ((ops.power_scalar((x - mean),2)).sum((0,))/x.shape[0]).reshape((1,x.shape[1])).broadcast_to(x.shape)
#             # 更新running mean和running var,用于在test使用
#             # mean_run 和mean是有区别的，mean_run没有做broadcast_to，不能对mean直接sum，因为broadcast_to之后的mean再做sum相当于增大了n倍。
#             mean_run = (x.sum((0,))/x.shape[0])
#             var_run = ((ops.power_scalar((x - mean),2)).sum((0,))/x.shape[0])
#             self.running_mean = (1-self.momentum)*self.running_mean + self.momentum*mean_run.data
#             self.running_var = (1-self.momentum)*self.running_var + self.momentum*var_run.data
#             norm = (x-mean)/ops.power_scalar((var+self.eps),(0.5))
#             return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
#         else:
#             norm = (x-self.running_mean)/ops.power_scalar((self.running_var+self.eps),(0.5))
#             return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
#         ### END YOUR SOLUTION
class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(dim, requires_grad=True))
        self.bias = Parameter(init.zeros(dim, requires_grad=True))
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            batch_mean = x.sum((0,)) / x.shape[0]
            batch_var = ((x - batch_mean.broadcast_to(x.shape))**2).sum((0,)) / x.shape[0]
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean.data
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var.data
            norm = (x - batch_mean.broadcast_to(x.shape)) / (batch_var.broadcast_to(x.shape) + self.eps)**0.5
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        else:
            norm = (x - self.running_mean.broadcast_to(x.shape)) / (self.running_var.broadcast_to(x.shape) + self.eps)**0.5
            return self.weight.broadcast_to(x.shape) * norm + self.bias.broadcast_to(x.shape)
        ### END YOUR SOLUTION



class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight =Parameter(init.ones(dim,device = device,dtype = dtype))
        self.bias = Parameter(init.zeros(dim,device = device,dtype = dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # 首先对整个mini-batch计算均值和方差
        # 假设是一个2D的tensor，第一个维度是batch size，第二个维度是特征维度
        # 经过summation后，得到的是一个1D的tensor，长度是特征维度
        # 所以要先reshape加一个维度，然后再broadcast_to
        # 这里的mean等价于 np.mean(x, axis=1,keepdims=True)
        mean = (x.sum((1,))/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # 有点没看懂
        var = (ops.power_scalar((x - mean),(2)).sum((1,))/x.shape[1]).reshape((x.shape[0], 1)).broadcast_to(x.shape)
        # 计算LN更新之后的值
        x_hat = (x-mean)/ops.power_scalar((var+self.eps),(0.5))
        y = self.weight.broadcast_to(x.shape)*x_hat + self.bias.broadcast_to(x.shape)
        return y
        
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.divide_scalar(ops.multiply(x,init.randb(*(x.shape),p=self.p)),(1-self.p))
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
    
        ### END YOUR SOLUTION
