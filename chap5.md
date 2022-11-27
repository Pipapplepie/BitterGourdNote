# 神经元（Neurons）与神经网络（Neural Netwok）

## M-P 神经元
（用于模拟生物神经元的数学模型）

$y = f(\Sigma_{i=1}^n w_i x_i - \theta) = f(w^T x + b)$

where $x \in \mathbb{R}^n$ （n个输入），经过w计算加权和。然后于自身阈值 $\theta$作比较，最后通过激活函数f（模拟“抑制”或“输出”），得到输出。

* 单个M-P神经元：感知机（sgn 阶跃函数 作激活函数）、对数几率回归（Sigmoid)

* 多个M-P神经元：神经网络

## 感知机

sgn 阶跃函数 作激活函数的神经元： $y = sgn(w^T x + b)$ (分类模型)。大于等于0取1；反之取0。

给定一个线性可分的数据集T，感知机的目的是求得能对数据集T中的政府样本完全正确划分的超平面，其中 $w^T x + b$为超平面方程。

超平面： $w^T x + b = 0$

* 超平面不唯一；
*  法向量w垂直于超平面；
*  法向量w和位移项b确定唯一一个超平面；
*  法向量指向的一半空间为正空间，另一半为负空间。

### 感知机学习策略

随机初始化w, b，将全体样本代入找出误差样本，归为子集$M \subset T&.

Then $\forall (x,y) \in M$, output denoted $\hat{y}$, we always have $(\hat{y} - y)(w^T x + b) \geq 0$ .

损失函数定义为： $L(w,b) = \Sigma_{x \in M} (\hat{y} - y)(w^T x + b)$

$\rightarrow w*, b* = arg \ min_{w,b} \ \Sigma_{x \in M} (\hat{y} - y)(w^T x + b)$

方法：随机梯度下降（可自行查阅）

## 神经网络

感知机只能扽类线性可分的数据集，对于线性不可分的数据集则无能为力，我们可以使用多个神经元构成的神经网络。

