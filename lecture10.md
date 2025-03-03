# convolution network

## convolutional operators in deep networks

太简单

## Elements of practical convolutions

### group convolution

![image-20240304172824082](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240304172824082.png)

如果有很多输入输出channel，那么计算量还是会很大，所以提出了group convolution。group convolution中output channel只取决input channel中的某几层而不是全部，最极限的状态就是只依赖一层，也就是output channel[0] = input channel[0]@weight

### dilations(扩张)

![image-20240304173633570](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240304173633570.png)

convolution有比较狭隘的感受野，3*3的kernel只能感受图片上3\*3大小的内容，所以Dilation出现了。

## differentiating convolutions

最好将convolution实现为needle的ops，而不是module。（好像是因为节约计算图的memory）所以我们需要实现convolution的正向计算compute和梯度计算gradient，adjoin。

![image-20240304174640178](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240304174640178.png)

术语：v_hat被称为adjoin，右边的偏导数被称为partial derivatives。

 ![image-20240304185346112](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240304185346112.png)

我们可以将convolution看作很多个matrix-vector product，所以我们先看matrix-vector product的求导结果。matrix-vector product的求导就是W的转置，所以backwards的时候只需要adjoint*W^T^即可。

那么问题是，convolution的转置是什么？

### Convolutions as matrix multiplication：version1（很有趣）

![image-20240304190226586](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240304190226586.png)

首先来看1D卷积的情况，我们将1D卷积的情况用matrix multiplication来表示的话，就如上面所示，W_hat根据filter w定义。我们并不会真这么计算，只是为了引出该怎么计算convolution的转置

![image-20240304190708731](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240304190708731.png)

那么我们怎么计算W_hat的transpose呢？将W_hat^T^写出来后发现，W_hat^T^和将filter flip之后做convolution的W_hat是一样的。也就是说W_hat^T^*V 和 将filter flip之后的filter:[w3,w2,w1]与v做convoluting结果是一样的，也就是说求adjoint operator只需要convolving v_hat（上层传过来的梯度）和flipped W。

### Convolutions as matrix multiplication：version2（很有趣）



![image-20240304191721631](https://gitee.com/zhou-lu-wu-bei/picture-hub/raw/master/image-20240304191721631.png)