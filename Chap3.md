# 一元线性回归（Univariate Linear Regression）

Model: $y = wx + b$

## 问题类型

**连续值问题**：e.g., 计算机水平 v. 发际线高度

**离散问题**： e.g., 颜值：好看（1）；不好看（0）

其中，离散问题有：

a. 有序的： e.g., 饭量（小：1；中：2；大：3）

b. 无序的： e.g., 肤色（黄：[1,0,0]; 黑: [0,1,0]; 白: [0,0,1]）

## 最小二乘法

Objective: 使均方误差最小,
i.e., minimize $E(w,b) = \Sigma_{i = 1}^m (y_i - f(x_i))^2$. 

## 极大似然估计(Maximal Likelihood Approximation)

$L(X, \theta) = \prod_{i = 1}^n \ P(x_1; \theta)$

P为X在情况 $\theta$ 下发生事件的概率。

X: 随机变量 $x_1, x_2, ...$; 

$\theta$为未知量，因此以上概率为一个关于 $\theta$ 的函数。

$\theta^* = arg min_{\theta} \ L(\theta)$

e.g., 设X: i.i.d. ~ $N(\mu, \sigma^2)$, 

$\therefore L(X, \theta) = L(X, \mu, \sigma^2)$.

### 对数函数：单调递增

$\ln L(\mu, \sigma^2)$ 与 $L(\mu, \sigma^2)$ **在同一处取最大值**，故有时我们会利用ln函数简化计算。

# 多元线性回归 (Multivariate Linear Regression)

# 对数几率回归 (Logistic Regression)

# 二分类线性判别分析 (Bisective Classification)
