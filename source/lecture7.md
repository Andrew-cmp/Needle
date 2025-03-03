# lecture7

## Programming abstractions

这一章主要学习框架的编程抽象，编程抽象是指 在框架内以什么样的方式来实现，扩展，执行model。

我们的目标就是学习他们的，实现自己的。

### 1. Forward and backward layer interface

![image-20240301165200251](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301165200251.png)

用forword和backword来定义**节点layer**的计算过程。所以layer在caffe中是basic elements，在这之中定义两个key function。反向传播和更新都是用backward函数来一层一层的更新weight和传播gradient。

这是一种比较自然的想法和选择，但只是第一代。

### 2. Computational graph and declarative programming(计算图和声明式编程)

![image-20240301173049592](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301173049592.png)

先定义好计算图，然后通过feed input来执行这张图。

这时候有了计算图的概念instead of layer，layer和computation graph的区别在于computation graph只是简单的描述前向计算的过程。要想创建computational graph，就要创建一个一个的node（占位符）。粉色框内是不进行计算的，只是对计算图的描述，下面的色sess.run()才执行计算。

优点：描述和执行是分开的，而且在执行之前一定已经知道了计算图的结构，所以在执行前能对计算图优化。比如V4可以替换成V3，以便于只对V3下的网络进行分析。另一点是描述和执行是分开的，执行可以在远端执行，针对远端的设备进行独立于计算图构建的设置。但在debug时，比如想看看v2的值print(v2)，这时是不能看的，因为还没执行。

### 3. Imperative automatic differentiation(命令式自动微分)

![image-20240301165654948](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301165654948.png)

这个和计算图构建很像，有什么区别呢？

计算图中，构建和运行是分开的，这里声明既计算。在我们构建计算图的时候就执行计算，并且允许python 简单的控制流，引入了动态计算图。

优缺点：只讨论optimization opportunities时，第2种更好一些，比如一些计算图节点的融合，或者计算图某一部分的计算，都是单声明更好优化。那第三种为什么好呢，因为这种对用户友好（第2种可能会自定一些语法或关键字tf.if等，而且允许一些python原生语言）。也更容易debug，因为计算是即时的。

所以第二种和第三种就是optimization和可用性之间的均衡。

## Elements of maching learning

1. The hypothesis class(模型）
2. The loss function（损失函数）
3. An optimization method（优化方法）
4. initialization(参数初始化)
5. regularization（正则化）
6. data loader and preprocessing（数据加载）

maching learning system本身就是一个非常自然的模块化系统。

### The hypothesis class(模型）

![image-20240301201110862](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301201110862.png)

这里以Residual net为例，block可以视为module，linear和ReLU也可以被视为module

![image-20240301201832636](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301201832636.png)

抽象出的module只要应该具备的事情

+ 至少应该是tensors in tensors out，

+ 怎么计算？

+ 给出可以被可以被训练的parameters

+ parameters初始化

  

### The loss function（损失函数）

**loss function是一种特殊的module**

![image-20240301202142030](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301202142030.png)

### An optimization method（优化方法）

![image-20240301202411685](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301202411685.png)

+ 通常需要model query作为input，包括trainable parameters
+ 追踪auxiliary states。

### Regularization

![image-20240301202706041](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301202706041.png)

正则化有两种方式

+ 作为loss function的一部分
+ 或者作为optimizer update的一部分，比如ppt下面的SGD

### initialization(参数初始化)

![image-20240301202926758](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301202926758.png)

### Data loader and preprocessing

![image-20240301203141663](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301203141663.png)

### 各个模块之间的合作图

![image-20240301203549705](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301203549705.png)