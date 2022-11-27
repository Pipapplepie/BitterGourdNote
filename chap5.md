# 神经网络（Neurasl Netwok）

## M-P 神经元
（用于模拟生物神经元的数学模型）

$y = f(\Sigma_{i=1}^n w_i x_i - \theta) = f(w^T x + b)$

where $x \in \mathbb{R}^n$ （n个输入），经过w计算加权和。然后于自身阈值 $\theta$作比较，最后通过激活函数f（模拟“抑制”或“输出”），得到输出。

* 单个M-P神经元：感知机（sgn 阶跃函数 激活函数）、对数几率回归（Sigmoid)

* 多个M-P神经元：神经网络
