"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        ### 想好sgd是在干什么
        # 整个的流程大概是：
        # 一、正向计算：计算出各个node的output和loss，
        # 二、反向传播：计算出各个node中param的grad，
        # 三、更新参数：将所有的param送入优化器optimalizer中，根据第二部中计算出的grad进行更新，也就是说第二部和第三步可以是解耦合的，二者只有数据依赖
        for param in self.params:
            if param.grad is None:
                continue
            else:
                if param not in self.u:
                    self.u[param] = ndl.zeros_like(param.data)

                # self.u[param].data = self.momentum*self.u[param].data + (1-self.momentum)*param.grad
                # param.data = (1-self.lr*self.weight_decay)*param.data - self.lr*self.u[param].data
                
                
                #将weight_decay前置到 加入到grad中以进行处理，而不是在更新param.data的时候处理
                if self.weight_decay > 0:
                    grad = param.grad.data + self.weight_decay * param.data
                else:
                    grad = param.grad.data
                self.u[param].data = self.momentum*self.u[param].data + (1-self.momentum)*grad
                param.data = param.data - self.lr*self.u[param]
                
        ### END YOUR SOLUTION

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        # 操了，之前把self.t的自增写下面去了，导致t的自增是每进来一个参数就增加而不是每一次step增加
        for param in self.params:
            if param not in self.m:
                self.m[param] = ndl.zeros_like(param.data)
            if param not in self.v:
                self.v[param] = ndl.zeros_like(param.data)
            grad = ndl.Tensor(param.grad, dtype='float32').data + param.data * self.weight_decay
                
            self.m[param] = self.beta1*self.m[param] + (1-self.beta1)*grad
            self.v[param] = self.beta2*self.v[param] + (1-self.beta2)*(grad**2)
            m_hat = self.m[param]/(1-self.beta1**self.t)
            v_hat = self.v[param]/(1-self.beta2**self.t)
            param.data = param.data - self.lr*m_hat/(v_hat**0.5+self.eps)
        
        ### END YOUR SOLUTION
