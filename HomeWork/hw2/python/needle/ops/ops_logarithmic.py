from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # input:(m,n),axes=(1,)
        # output:(m)
        max_z_original  =array_api.max(Z,axis=self.axes,keepdims=True)
        max_z_reduce = array_api.max(Z, self.axes)
        e = array_api.exp(Z-max_z_original )
        log = array_api.log(array_api.sum(e,axis=self.axes))
        return max_z_reduce + log
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
            
        #不懂，好像是求倒是把max直接忽略了
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        # 对log(x)的求导
        grad_sum_exp_z = out_grad / sum_exp_z
        # 对sunmation(x)的求导
        grad_exp_z = grad_sum_exp_z.reshape(new_shape).broadcast_to(z.shape)
        # 对exp(x)的求导
        return grad_exp_z * exp_z
            
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

