# lecture9

## initialization

![image-20240302145550098](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302145550098.png)

n是x的Dimension number，activation normal= ||Z||^2^，gradient normal=||▽~wi~loss||^2^。

这一页介绍了Wi不同初始化范围的影响。当c=2时，对ReLU network来说工作的很好，activation norm很平稳的在10^1^，但c=1或c=3时，activation norm就会逐渐变大或者变小，这会导致对应的梯度值也会过小或者过大。这很不好，比如c=3时，activation过大，会导致loss值变为NaN，overflow，并且在backwards时，过大的gradient会导致更新weight变得不稳定。c=1时，过小gradient会导致无法进行梯度下降，也就是No progress。

按照凸优化的理论，初始值选在哪并不重要，反正最后都会到极值点。但in practice，在NN中，不好的initialition会导致根本没法训练，无论多深，无论怎么训练。



![image-20240302151227488](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302151227488.png)

**即使训练成功，初始化时出现的效果/尺度也会在整个训练过程中持续影响。**

我们对这个DNN进行训练，得到了下面三张图，可以看到前两张图虽然初始值还是对其有影响，但都相对于初始值变化了很多，而最后一张图告诉我们，**初始化的weight变化的很小**

所以一个更好的initialization会帮助更好的optimization，这就是标题中initialization vs.optimization.

## Normalization

initialization很重要，为什么重要，因为不同的initialization，会导致the norm of activations 变得不再一样。所以我们可以直接在网络结构中改变the norm of activation。

![image-20240302152114040](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302152114040.png)

初始化对于训练非常重要，并且在训练过程中可能会发生变化，从而在各层/网络之间不再“一致”，layer可以做任何计算上的事情，包括normalization，所以我们可以在网络中加入 the normalization of the activations的层。

### layer normalization

![image-20240302152443706](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302152443706.png)

所以layer normalization出现。强迫activations的mean=0，variance=1.

![image-20240302152827294](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302152827294.png)

layer normalization解决了前几个ppt中出现的问题，activation norm保持稳定了，gradient也改变了值随weight初始化的值而改变的这一个问题。

### batch normalizaiton

![image-20240302153512492](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302153512492.png)

![image-20240302154324233](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302154324233.png)

听不懂，不管了。

## regularization

![image-20240302154806265](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302154806265.png)

这一页介绍了regularization的motivation，神经网络基本都是overfit，缺少泛化能力。

通常，深层网络（甚至您在作业中写的简单的两层网络）都是过度参数化的模型：它们包含的参数（权重）比培训示例的数量更多。这意味着（正式地，在一些假设下），它们能够准确地拟合训练数据。在“传统”ML/统计思维中（有一些重大警告），这应该意味着模型将过度拟合训练集，并且不能很好地概括 • ...但是它们确实可以很好地概括测试示例 • ...但并非总是如此（许多较大的模型通常仍然会过度拟合）

![image-20240302155027854](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302155027854.png)



正则化是“限制function的复杂性”的过程，以确保网络的泛化能力；

通常以两种方式出现在深度学习中，

+ 隐式正规化是指通过我们现有算法（即SGD）或体系结构来限制了所考虑的功能的方式
  + 例如，我们实际上并未对“所有神经网络”进行优化，我们正在优化所有优化SGD考虑的神经网络，具有给定的重量初始化
+ 明确的正则化是指对网络进行的修改和训练过程明确旨在正规化网络

### L2正则化

![image-20240302155558527](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302155558527.png)

![image-20240302155736067](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302155736067.png)

这里三种都有相同的generation loss，所以参数大小可能不是很好衡量模型复杂度的工具。

## dropout

dropput是什么

![image-20240302155911816](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302155911816.png)

dropout的作用原理

![image-20240302160008304](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302160008304.png)

后面都是鸡汤。