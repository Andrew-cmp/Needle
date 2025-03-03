"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

import numpy as array_api

## 继承自tensorop
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a+b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

##辅助函数 EWiseAdd()意味着一个EwiseAdd实例，
def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar                                   
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        grad_a = out_grad*(self.scalar)*(a**(self.scalar-1))
        return grad_a
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION

        return array_api.divide(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # ### BEGIN YOUR SOLUTION
        # if not isinstance(node.inputs[0], NDArray) or not isinstance(
        #     node.inputs[1], NDArray
        # ):
        #     raise ValueError("Both inputs must be tensors (NDArray).")
        a, b = node.inputs[0], node.inputs[1]
        grad_a = array_api.divide(out_grad,b)
        grad_b = array_api.negative(array_api.multiply(a,array_api.divide(out_grad,array_api.multiply(b,b))))
        return grad_a,grad_b
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        
        #print(type(out_grad/self.scalar))
        #print(type(array_api.divide(out_grad,self.scalar)))
        #return array_api.divide(out_grad,self.scalar)???????????????好像有点明白了，虽然在计算过程中是可以使用numpy的运算，但是最后还是要返回一个tensor，所以要用tensor的api
        #autograd的gradient_as_tuple函数里面也没有检测是不是array，所以这里返回的梯度是不能为array的
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        axes = self.axes
        ### BEGIN YOUR SOLUTION
        if(self.axes is None):
            ## 草，默认为最后两个轴，和np.transpose的默认不一样
            ndim = a.ndim
            axis = array_api.arange(a.ndim)
            
            axis[ndim-1], axis[ndim-2] = axis[ndim-2],axis[ndim-1]
            return array_api.transpose(a,axis)
        else:
            #print(self.axes)
            #print(array_api.array(self.axes))
            #这个地方要注意，输入的是要交换的数组维度，但numpy.transpose的输入是新数组的维度。
            axis = array_api.arange(a.ndim)
            #交换值https://stackoverflow.com/questions/22847410/swap-two-values-in-a-numpy-array
            axis[axes[0]], axis[axes[1]] = axis[axes[1]],axis[axes[0]]
            return array_api.transpose(a,axis)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        #return array_api.reshape(out_grad,node.inputs[0].shape) error?       
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        
        return array_api.broadcast_to(a,self.shape)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        
        
        ### BEGIN YOUR SOLUTION
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]
        for i, (ori, cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori == cur:
                shrink_dims[len(self.shape) - i - 1] = -1
        shrink_dims = tuple(filter(lambda x: x >= 0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(ori_shape)
        ### END YOUR SOLUTION
        
        # ### BEGIN YOUR SOLUTION
        # ##这里是输入b的shape，也是将要return的梯度值的shape
        # origin_shape = node.inputs[0].shape
        # ##输出b_expand的shape,也是out_grad的shape
        # output_shape = out_grad.shape
        # #需要求和的维度是哪几个。这个维度针对的是out_grad中的维度，算出是哪几个维度，用sum一计算就ok了
        # expand_e = []
        # #需要从后往前算，因为当维度缺失时，是从后往前扩展的
        # for i in range(-1,-len(output_shape)-1,-1):
        #         # origin_shape的长度可能会比output_shape短，
        #         # 比如origin_shape=(1,)，output_shapee=(1,2)。
        #     if(-i>len(origin_shape)):
        #         expand_e.append(i)
        #         continue
        #     # 如果目标维度和输出维度不一样，那么就需要扩展
        #     ## 比如origin_shape=(1,2)，output_shapee=(4,2)。
        #     if(origin_shape[i] != output_shape[i]):
        #         expand_e.append(i)
                
        # # out_grad进行sum运算，运算的轴axes是b_exp相对于b经过拓展的维度
        # ans = summation(out_grad,expand_e)
        
        # # 因为res的形状可能与lhs(也就是b)不相同，所以这里需要reshape到b原本的维度上。这种情况一般出现在b的维度比较少的时候。
        # # 比如b.shape=(4,)，b_expand.shape=(4,2),此时即便b_expand求和完毕后也是（4，1)的shape。
        # return reshape(ans,origin_shape)
        
        # ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return array_api.sum(a,self.axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        if isinstance(self.axes,Number):
            axes = (self.axes,)
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION

# MatMul需要考虑两个形状不一致时的情况
# 比如（4，5，2，3）@（3，2）
# 这个时候其实就是每次计算（2，3）@（3，2）的结果，一共计算4*5次，然后再把结果拼接起来，最终的结果就是（4，5，2，2）
    def gradient(self, out_grad, node):
        # 假设lhs.shape=(4,5,2,3)，rhs.shape=(3,2)，out_grad.shape=(4,5,2,2)
        # 那么lgrad.shape=(4,5,2,3)，rgrad.shape=(3,2)
        
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        # 此时lgrad.shape=(4,5,2,3)，rgrad.shape=(4,5,3,2),rgard的形状不对，需要对其处理
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        # 走这条道路的时候，rhs.shape=(3,2)，rgrad.shape=(4,5,3,2)，所以需要对，rgrad进行sum运算
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return - out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0,a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data().copy()
        out[out > 0] = 1
        return out_grad * Tensor(out)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)
