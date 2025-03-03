# lecture8

## Needle Refresh

### Mutating the data field of a needle Tensor（改变needle tensor的值域）

#### detach

![image-20240301211652108](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301211652108.png)

这种实现方法就有问题，我们将某个变量的所有历史更新都存起来了，这回导致内存和运行速度的问题，事实上我们只需要记录每个节点的前一个节点就可以了。

所以我们提出了“data”关键字。

![image-20240301211815823](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301211815823.png)





这里是通过tensor类的detach()方法来将w.data从计算图里摘出来的。新建一个tensor，和w共享underlying cached data，所以w.data和w共享memory，但是我们移除了所有其他的fields比如inputs field和op field，这是他看起来就是一个常数，并且requires_grad设置为false，变成不需要梯度的常量constant

![image-20240301215314907](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301215314907.png)

![image-20240301215717549](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301215717549.png)

这里是因为w.data和grad.data都被detach了，所以计算出的new_w也被detach了，没有梯度和input。所以当一个tensor的所有input requires_grad都为false，这个tensor的requires_grad也为false

![image-20240301215936992](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301215936992.png)

**总结一下**，为什么detach好呢？因为如果构建计算图时，节点都hold了之前的computation graph node，这些hold 的node都不会被released导致内存爆炸。

### Numerical Stability

老生常谈的float的精度问题和取值范围的问题。这里是拿softmax种exp(100)举例子的

![image-20240301221640038](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301221640038.png)

怎么解决的呢，这关于softmax的特性，就是输入同时减去一个值c，输出不变。比如softmax([10,10,11])和softmax([0,0,-1])的结果是一样的，同样的性质也适用于logsoftmax和logsumsoft等。大部分框架都会自动检测这种overflow然后re-normalization

![image-20240301222004905](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301222004905.png)

## Designing a Neural Network Library（设计一个NN库）

### nn.Module interface

第一个出场的是parameter，继承自tensor。他没有什么特殊的实现，只是告诉别人：“嗨这是一个tensor called parameter”

![image-20240301222856391](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240301222856391.png)

![image-20240302131727293](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302131727293.png)

![image-20240302133059640](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302133059640.png)

scaleAdd定义了一个具体的module：y=x*s+b，其中self.s和self.b是parameter。

下面这个地方的调用.parameters()，紧接着调用__get_getparm()，递归的将所有parameter返回。self.\_\_dict\_\_就是{'s':s，‘b’:b}（其中key一定是名字，所以拿value出来），并且已经被定义为了parameters.

![image-20240302133955331](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302133955331.png)

虽然这里没有包含任何parameter，但包含了多个submodule。这就像是pytorch中将所有的module或layer的parameter遍历出来。

## Loss function

![image-20240302134352801](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302134352801.png)

Loss function可以被实现为module，这个module不包含任何parameters。

![image-20240302140337592](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302140337592.png)

这里params[0].grad和mpath.path0.s.grad是一样的，下面w的反向更新，由于使用的w.data来更新，所以实际上只修改了w的值域，没有对其他域修改，所以之后可以方便的在进行forward使用和backward更新。

## optimizer

![image-20240302140846137](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302140846137.png)

接下来定义了Optimizer的两个接口：reset_grad用来重置梯度，step用来更新parameter。这也是optimizer的两个用处

![image-20240302141027430](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302141027430.png)

这就很像pytorch的训练方式了。python的语法很帅，相当于opt对象存储了指向parameter的指针而不是parameter的值，所以通过opt更新，同一个memory区域的值就更新了，module的值也更新了。

## initialization

![image-20240302141918664](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302141918664.png)

待实现

## Additional contents on programming model

可以引入很多新的特性和内容。以TensorTuple为例子

### fused operator and tuple value

到目前为止，每个 Needle 运算符仅返回一个输出张量。在现实世界的应用场景中，有时在单个（融合后的）运算符中同时计算多个输出会很有帮助。这就是算子融合。
Needle 旨在支持此功能。为此，我们需要引入一种新的值——tuple。

![image-20240302142127563](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302142127563.png)

![image-20240302142511090](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302142511090.png)

![image-20240302142706239](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240302142706239.png)

这里相当于框架原生支持fusedaddsaclars这个融合算子，所以可以直接用。