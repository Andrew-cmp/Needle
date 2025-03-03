	这次作业的目的是建立一个自动差分框架，然后使用这个框架重新实现hw0中简单的两层神经网络。

## introduction to needle

​	needle库中有两个重要文件，即python/pine/autograd.py文件，定义了计算图框架的基础，也将构成自动差分框架的基础，以及python/needle/ops/ops_mathematic.py.file，其中包含各种运算符的实现，您将在整个作业和课程中使用这些运算符来实现。

​	尽管autograd.py文件中已经写好了了自动微分的基本框架，但您应该熟悉库的基本概念，因为它与几个定义的类有关。请注意，我们不建议在开始实现之前尝试通读整个代码库（在实现某些功能后，某些功能可能会更有意义），但您应该有一个基本的。特别是，您应该熟悉以下类背后的基本概念：

+ Value：在计算图中计算的值，即应用于其他Value对象的运算的输出，或常量（叶子）Value对象。我们在这里使用一个泛型类（然后我们专门化为例如张量），以便在之后版本的needle中可以允许我们使用其他数据结构，但目前将主要通过它的子类Tensor与该类交互。
+ Op：计算图中的运算符。运**算符需要在compute（）方法中定义其“正向”传递（即，如何在Value对象的基础数据上计算运算符），以及通过gradient（）方法定义其“反向”传递（定义如何与传入的输出梯度相乘）**。编写此类运算符的详细信息将在下面给出。
+ Tensor：这是Value的一个子类，对应于实际的Tensor输出，即计算图中的多维数组。此作业的所有代码（以及以下大部分代码）都将使用Value的这个子类，而不是上面的泛型类。我们提供了几个convevience function（例如，运算符重载），允许您使用正常的Python约定对张量进行操作，但在实现相应的操作之前，这些函数将无法正常工作。
+ TensorOp：这是返回张量的运算符Op的subclass。作业中所有的operations都属于此类型。

## 第一题 Implementing forward computation 

```python
class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad

def add(a, b):
    return EWiseAdd()(a, b)
```


​	这个类的实现遵循以下约定。compute() 函数计算“前向”传递，即仅计算操作本身。这里注意，compute的输入都是NDArray对象（在此初始实现中，它们是numpy.ndarray对象，在以后的作业中需要实现自己的NDArray）。也就是说，compute() 在原始数据对象NDArray上进行前向传递，而不是在automatic differentiation中的Tensor对象上进行计算。

​	我们将在下一部分讨论gradient()调用，但在这里强调一下，**这个调用与forward不同，因为它接受Tensor参数。这意味着在这个函数内部进行的任何调用都应该通过TensorOp操作本身进行**（以便可以对梯度进行梯度）。

​	最后，请注意，我们还定义了一个辅助的add()函数，以避免调用EWiseAdd()(a,b)（这有点繁琐）时添加Tensor对象。这些函数都已为您编写，应该是自解释的。

​	对于这一题，您需要为以下每个类实现compute调用。这些调用非常直接，基本上应该是调用相关numpy函数的一行代码。请注意，由于在以后的任务中，您将使用除numpy之外的后端，我们已将numpy导入为import numpy as array_api，因此如果要使用典型的np.X()调用，您将需要调用array_api.add()等。

+ PowerScalar: 将输入值计算为到整数（标量）次幂
+ EWiseDiv: 逐元素除法，即对应元素相除（2个输入）
+ DivScalar: 对输入值进行逐元素除法，除数为标量（1个输入，标量 - 数字）
+ MatMul: 输入值的矩阵乘法（2个输入）
+ Summation: 沿指定轴对数组元素进行求和（1个输入，轴 - 元组）
+ BroadcastTo: 将数组广播到新的形状（1个输入，形状 - 元组）
+ Reshape: 为数组提供新的形状，不改变其数据（1个输入，形状 - 元组）
+ Negate: 数值的逐元素取负（1个输入）
+ Transpose: 反转两个轴的顺序（axis1，axis2），默认为最后两个轴（1个输入，轴 - 元组）

很简单就不放代码了

## 第二题Implementing backward computation [25 pts]

现在已经实现了计算图中的functions了，为了使用计算图模型实现自动微分，我们还需要计算反向传递过程，**也就是将输入的 backward gradients和the relevant derivatives of the function相乘。**

**最通用的方法就是，假设输入都是saclar，计算偏导数，然后再match size。**

The general goal of reverse mode autodifferentiation is to compute the gradient of some downstream function ℓ of f(x,y) with respect to x (or y). Written formally, we could write this as trying to compute

$$
\begin{equation}

\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x}.

\end{equation}
$$
**输入的 backward gradients就是这里的$ \frac{\partial \ell}{\partial f(x,y)}$,所以我们希望 `gradient()` functiont可以最终将我反向传播来的梯度和我们函数自己的导数 $\frac{\partial f(x,y)}{\partial x}$相乘**。

这里举了ElemwiseAdd和ElemwiseMul两个例子。

----

To see how this works a bit more concretely, consider the elementwise addition function we presented above
$$
\begin{equation}

f(x,y) = x + y.

\end{equation}
$$
Let's suppose that in this setting $x,y\in \mathbb{R}^n$, so that $f(x,y) \in \mathbb{R}^n$ as well.  Then via simple differentiation
$$
\begin{equation}

\frac{\partial f(x,y)}{\partial x} = 1

\end{equation}
$$
so that
$$
\begin{equation}

\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial \ell}{\partial f(x,y)}

\end{equation}
$$
i.e., the product of the function's derivative with respect to its first argument $x$ is just exactly the same as the backward incoming gradient.  The same is true of the gradient with respect to the second argument $y$.  This is precisely what is captured by the following method of the `EWiseAdd` operator.

```python
  def gradient(self, out_grad: Tensor, node: Tensor):

​    return out_grad, out_grad
```

i.e., the function just results the incoming backward gradient (which actually *_is_* here the product between the backward incoming gradient and the derivative with respect to each argument of the function.  And because the size of $f(x,y)$ is the same as the size of both $x$ and $y$, we don't even need to worry about dimensions here.

----

Now consider another example, the (elementwise) multiplication function
$$
\begin{equation}

f(x,y) = x \circ y

\end{equation}
$$
where $\circ$ denotes elementwise multiplication between $x$ and $y$.  The partial of this function is given by
$$
\begin{equation}

\frac{\partial f(x,y)}{\partial x} = y

\end{equation}
$$
and similarly
$$
\begin{equation}

\frac{\partial f(x,y)}{\partial y} = x

\end{equation}
$$
Thus t compute the product of the incoming gradient
$$
\begin{equation}

\frac{\partial \ell}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \frac{\partial f(x,y)}{\partial x} = \frac{\partial \ell}{\partial f(x,y)} \cdot y

\end{equation}
$$
If $x,y \in \mathbb{R}^n$ like in the previous example, then $f(x,y) \in \mathbb{R}^n$ as well so the first element returned back the graident function would just be the elementwise multiplication
$$
\begin{equation}

\frac{\partial \ell}{\partial f(x,y)} \circ y

\end{equation}
$$
This is captures in the `gradient()` call of the `EWiseMul` class.

```pyt
class EWiseMul(TensorOp):

  def compute(self, a: NDArray, b: NDArray):

​    return a * b

  def gradient(self, out_grad: Tensor, node: Tensor):

​    lhs, rhs = node.inputs

​    return out_grad * rhs, out_grad * lhs
```

**Hint**:**加法乘法的计算过程可能很直观，但Broadcast和Summation这种函数的导数就不是很明显**，为了更好地理解这些内容，你可以查看这些数值的实际数值，并输出它们的实际值，如果你不知道从哪里开始的话（请参阅 tests/test_autograd_hw.py，特别是该文件中的 check_gradients() 函数，以了解如何执行这个操作）。**记住，`out_grad` 的大小始终是操作的输出大小，而 `gradient()` 返回的 Tensor 对象的大小必须始终与操作的****原始输入相同。**

### broadcast算子

作者：方鸿渐
链接：https://www.zhihu.com/question/561694502/answer/3050230031
来源：知乎
著作权归作者所有。商业转载请联系作者获得授权，非商业转载请注明出处。

## broadcast_to算子

我们考虑一个非常简单的例子，画出下面的这张计算图。

![img](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/v2-65a54c04805e6_1440w.webp)

b_exp就是b经过broadcast_to算计得到的结果。上图中红色字体表示这是一个算子。图中的算子有广播算子、矩阵加法算子以及求和算子。

我们首先计算d相对于c的梯度：

![img](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/v2e5fd4dee762850af_1440w.webp)

很多同学可能对于怎么计算感觉到很头疼，因为分子c是一个张量。其实我们如果换一个角度思考这件事情，就会发现特别简单。d是在c上执行sum运算得到的，这也就意味着d = c[0] + c[1] ，所以为了计算d对c导数，我们可以考虑分别计算d相对于c[0]和c[1]的导数。因此，可以有上面那张图片的过程，也就是:

![image-20231206184520057](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20231206184520057.png)

可以说，这种将一个张量拆分成若干个标量来思考的方式是理解复杂算子的反向传播的一大利器。

接下来继续计算d相对于a的梯度：

![img](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/v28b2ed1749b6cfc_1440w.webp)

然后计算d相对于b_exp的梯度：

![img](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/aedfac4fb5b1e_1440w.webp)

好！接下来就是我们的重点，关于如何计算d关于b的梯度了。其实答案非常的简单，我们只需要在b_exp上进行sum运算就可以了。也就是说，b的梯度实际上是等于b_exp在b扩展的轴上进行求和的结果。因此，b的梯度实际上就是[1+1]=[2]。那为什么会这样子呢？

实际上我们可以这样子思考，d关于b_exp的的梯度b_exp_grad是[1, 1]，又因为b_exp的元素是由b的元素经过“复制粘贴”而来的，所以说b_exp_grad[0]是b[0]的梯度，而且b_exp_grad[1]也是b[1]的梯度。因此，b的梯度b_grad = b_exp_grad[0] + b_exp_grad[1]。

我们可以用下面的图更加直观的理解：

![img](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/v2-1b399b6dd3f3683b5b4d39e1110af196_1440w.webp)

黑色字体表示这是张量a的元素；橙色表示这个张量b的元素；红色表示这是张量c的元素；绿色表示这是张量d的值；棕色标志这是一个算子。

上面这个图其实表示的就是我们刚才讨论的运算。在这个图中，我们可以看到b中的唯一个的元素被用了两次。第一次是用在和1(也就是c[0])相加，第二次使用在和2(也就是c[1]相加)，这其实就是广播的本质：**用同一个数和多个不同的数进行运算**。因此，求导的过程如下：

![img](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/v2-0ddee42dce955fbdc22b1d26bd0c032c_1440w.webp)

篮色字体表示这是对应数字的梯度

从上面这种计算导数的图中可以看到，3(也就是b[0])的梯度是2。因为b有两条路径可以到达9(也就是d)，说明d被用了两次，根据[链式求导法则](https://www.zhihu.com/search?q=链式求导法则&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3050230031})，b[0]的梯度应该是两条路径上的梯度的和。

记住：**[广播机制](https://www.zhihu.com/search?q=广播机制&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3050230031})所作其实就是让同一个数(比如b[0]=3)多个不同的数(比如a[0]=1和a[1]=2)进行运算。**

于是，广播算子broadcast_to的反向传播的实现可以使用下面的算法准确描述：

```python3
class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape. # 想要广播到目的形状

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        # out_grad是经过维度拓展之后的张量的梯度，也就是b_exp的梯度b_exp_grad
        # node是张量b_exp，node.inputs[0]就是b。
        lhs = node.inputs[0]  # 算子的输入，也就是b=[3]
        origin_shape = lhs.shape  # 原本的形状，也就是b.shape=(1,)
        target_shape = self.shape  # 想要变换到的形状，也就是b_exp.shape=(1,2)
        expanded_axes = []  # 记录哪一个维度被拓展了
        for i in range(-1, -len(target_shape)-1, -1):  # 从尾部开始遍历
            if i < -len(origin_shape):
                # origin_shape的长度可能会比target_shape短，
                # 比如origin_shape=(1,)，target_shape=(1,2)。
                expanded_axes.append(i+len(target_shape))
                continue
            if target_shape[i] != origin_shape[i]:
                # 如果目标形状与原本的形状不相同
                # 那就说明这个维度经过了拓展，需要记录到expanded_axes中
                expanded_axes.append(i + len(target_shape))
        # out_grad进行sum运算，运算的轴axes是b_exp相对于b经过拓展的维度
        res = summation(out_grad, tuple(expanded_axes))
        # 因为res的形状可能与lhs(也就是b)不相同，所以这里需要reshape到b原本的维度上。
        res = reshape(res, origin_shape)
        return res
```

算法的主要步骤就是:

1. 定义broadcast_to算子的target_shape和origin_shape。origin_shape是输入张量的维度，也就是b.shape=(1,)。target_shape是输出张量的维度，也就是target_shape=(1,2)。
2. 定义expanded_axes列表为：对target_shape和origin_shape从后往前遍历，如果target_shape[i]不等于origin_shape[i]，则说明第i个维度经过经过拓展，需要放入到expanded_axes中。
3. 在expanded_axes指定的维度上进行求和。

算法的一些需要注意的点是:

1. 由于广播机制的要求，在使用[for循环](https://www.zhihu.com/search?q=for循环&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3050230031})匹配target_shape[i]和origin_shape[i]的时候，需要从后往前遍历。对于广播机制不理解的同学，可以在bing搜索一下广播机制。
2. origin_shape的长度可能会比target_shape短。
3. 最后需要对结果reshape一下。经过sum运算之后，得到梯度(在上面的代码中是[res张量](https://www.zhihu.com/search?q=res张量&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra={"sourceType"%3A"answer"%2C"sourceId"%3A3050230031}))的形状可能与原本的形状有所不同，比如原本的形状是(2,3,4)，而计算得到的梯度是(2,3,1,4)，因此需要进行一个reshape操作。

下面这张图说明了广播算子反向传播的过程。

![img](https://picx.zhimg.com/80/v2-7d8c284c1bbf09b3a4c9192b2210f55f_1440w.webp?source=1def8aca)

### sumation算子

sumation算子同样可以参考上面的博文，可以看作是broadcast算子的反方向，也是要对比input和output之间的求和情况，看看是哪几个维度被求和了。

```python
class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION

        return array_api.sum(a,self.axes)

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        # 此node的输入时b shape为（2，2，2，2），self.axes=（2，3），要返回的梯度也是这个shape
        # 此node的输出是（2，2），out_grad也是这个大小
        # ①要找到self.axes并让（2，2）变成（2，2，1，1）
        # ②对out_grad直接broadcast_to变成（2，2，2，2）
        ### BEGIN YOUR SOLUTION
        # 创建要返回的梯度的shape
        new_shape = list(node.inputs[0].shape)
        # 找到要拓展的轴是哪几个，如果调用时没给轴，那就是得到了一个标量，需要拓展的轴就是全部，否则就是给的self.axes的值
        axes = range(len(new_shape)) if self.axes is None else self.axes
        # 进行第一步
        for axis in axes:
            new_shape[axis] = 1
        # 进行第二步
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)
        ### END YOUR SOLUTION
```

### MatMul算子

```PYTHON
class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a,b)
        ### END YOUR SOLUTION
    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lgrad, rgrad = matmul(out_grad, rhs.transpose()), matmul(lhs.transpose(), out_grad)
        if len(lhs.shape) < len(lgrad.shape):
            lgrad = lgrad.sum(tuple([i for i in range(len(lgrad.shape) - len(lhs.shape))]))
        if len(rhs.shape) < len(rgrad.shape):
            rgrad = rgrad.sum(tuple([i for i in range(len(rgrad.shape) - len(rhs.shape))]))
        return lgrad, rgrad
        ### END YOUR SOLUTION
```

### reshape算子

### transpose算子

### ReLU算子

![image-20231206204119409](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20231206204119409.png)

```python
class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        array_api.maximum(0,a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out = node.realize_cached_data().copy()
        out[out > 0] = 1
        return out_grad * Tensor(out)
        ### END YOUR SOLUTION
```

## Question 3: Topological sort [20 pts]

给定一张图，找到其拓扑排序之后的list。

```python
def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION
```

## Question 4: Implementing reverse mode differentiation [25 pts]

​	做完拓扑排序就可以按照拓扑节点的序列来进行自动微分了。拓扑序列的最后一个点就是我们首先要计算的值，以此类推。

![image-20231206212806069](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20231206212806069.png)

​	![image-20231206210351347](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20231206210351347.png)

![image-20231206210405717](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20231206210405717.png)

总结一下，在以下代码中，`node_to_output_grads_list`中的元素为adjoint δy/δv~i~ 。但i节点的adjoint δy/δv~i~ 是来自于所有的出箭头指向的节点的，应该为所有出箭头指向的node的返给的梯度之和。如下面第22行代码，需要将所有反过来的梯度相加，这是才是这个节点关于y的偏导数adjoint。

```python
def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
        # 从后往前遍历，对每个node都要做一次
        # 已知这个node关于y的梯度
        # 可以计算得到这个node的input node关于y的梯度
        # 这个node的关于y的梯度是这个node的所有输出边的梯度的和
        sum_node_list(node_to_output_grads_list[node])
        output_grads = node_to_output_grads_list[node]
        # 这时是叶子节点，没有op
        if node.op is None:
            continue
        # 这个node的输入
        inputs = node.inputs
        # 这个node的op
        op = node.op
        # 这个node对每个输入的梯度
        input_grads = op.gradient_as_tuple(output_grads, node)
        # 对每个node的输入节点，把这个node的关于此节点的梯度加到这个节点的梯度列表里
        for i in range(len(inputs)):
            input_node = inputs[i]
            if input_node not in node_to_output_grads_list:
                node_to_output_grads_list[input_node] = []
            node_to_output_grads_list[input_node].append(input_grads[i])
  
    ### END YOUR SOLUTION
```

## Question 5: Softmax loss [10 pts]

我们将实现在homework0的question3中实现过的softmax函数，softmax loss takes as input a `Tensor` of logits and a `Tensor` of one hot encodings of the true labels.
$$
\begin{equation}

\ell_{\mathrm{softmax}}(z, y) = \log\sum_{i=1}^k \exp z_i - z_y.

\end{equation}
$$
最重要的是要将homework0中的操作写稿围殴needle中的object和operation。像前一个作业一样，要计算average softmax loss over a batch of size m。

**一定注意，这里的Z*y_one_hot,我们需要减去的是这个label的概率Z[y]，而不是减去这个label!!!!**

```python
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
    sum = ndl.sum(power, axis=1, keepdims=True)
    log = ndl.log(sum)
    loss = ndl.sum(log - Z * y_one_hot, axis=1)
    return ndl.mean(loss)
    ### END YOUR SOLUTION
```

## Question 6: SGD for a two-layer neural network [10 pts]

相比homework0，W1和W2都是Tensor，但input x和y都是numpy array，由于需要进行batch，所以需要将batch_X从numpy转化为Tensor，然后对y_batch进行独热编码之后转换为Tensor

```python
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

```

