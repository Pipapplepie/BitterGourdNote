# 机器学习三要素

模型；
策略；
算法；

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

_def._ $L(X, \theta) = \prod_{i = 1}^n \ P(x_1; \theta)$

P为X在情况 $\theta$ 下发生事件的概率。

X: 随机变量 $x_1, x_2, ...$; 

$\theta$为未知量，因此以上概率为一个关于 $\theta$ 的函数。

$\theta^* = arg min_{\theta} \ L(\theta)$

e.g., 设X: i.i.d. ~ $N(\mu, \sigma^2)$, 

$\therefore L(X, \theta) = L(X, \mu, \sigma^2)$.

### 对数函数：单调递增

$\ln L(\mu, \sigma^2)$ 与 $L(\mu, \sigma^2)$ **在同一处取最大值**，故有时我们会利用ln函数简化计算。

## Alternate: $y = wx + b + \epsilon$

where $\epsilon$ ~ $N(0, \sigma^0)$ 不受控制的随机误差.（中心极限定理：Central Limit Theorem）

概率密度函数 pdf for $\epsilon$：

$P(\epsilon) = \frac{1}{\sqrt{2\pi}\sigma} \exp ( - \frac{\epsilon^2}{2\sigma^2})$

$\rightarrow L(w,b) = \prod_{i = 1}^m \ p(y_i) = \prod_{i = 1}^m \ \frac{1}{\sqrt{2\pi}\sigma} \exp ( - \frac{[y_i - (wx_i + b)]^2}{2\sigma^2})$

$\rightarrow \ln L(w,b) = m \ln \frac{1}{\sqrt{2\pi}\sigma} + \sum_{n=1}^m ( - \frac{[y_i - (wx_i + b)]^2}{2\sigma^2})$

$\rightarrow (x*, b*) = arg max_{(w,b)} \ L(w,b) = arg max_{(w,b)} \ \sum_{n=1}^m [y_i - (wx_i + b)]^2$

**故，结论与最小二乘法相同！**

### 殊途同归

**_Claim:_** $E(w,b) = \Sigma_{i = 1}^m (y_i - f(x_i))^2$ is a convex function. （凸函数）

_def._ (of convex set) $D \subset \mathbb{R}^n$, $\forall x,y \ \in D$ and $\forall \alpha \in [0,1]$, we have
$\alpha x + (1 - \alpha)y \ \in D$.

_def._ (of convex function) D is a nonempty convex set, f is defined on D, for any $x^1$, $x^2 \ \in D$, $\forall \alpha \in [0,1]$, we have

$f(\alpha x^1 + (1 - \alpha)y) \leq \alpha f(x^1) + (1 - \alpha)f(x^2)$.

_def._ (Gradient)

_def._ (Hessian Matrix)

_def._ (Positive Semidefinite)

_theorem._ If the Hessian matrix of $E(w,b)$ is Positive Semidefinite, then $E(w,b)$ is a convex function with respect to w and b.

可证$E(w,b)$ 为凸函数。（skipped）

# 多元线性回归 (Multivariate Linear Regression)

最小二乘法 ————> 损失函数（Loss Function）

$f(x_i) = w^T  x_i + b$ where $x_i$, $w$, $b$ are vectors, say in $\mathbb{R}^n$

<img src='https://user-images.githubusercontent.com/107236740/202985560-906bda9f-a7f8-435f-8f12-b3b84292b942.png' width='50%'>

$\therefore$ The loss function is:

![en](https://user-images.githubusercontent.com/107236740/202981424-c2be835a-b7ac-46de-a11f-0fe456d9bd18.png)

**也可以将其简化为：** $E_{\hat{w}} = (y - X \hat{w})^T \cdot (y - X \hat{w})$

where $\hat{w} = [w^T b]^T$ and 

<img src='https://user-images.githubusercontent.com/107236740/202983042-28d33c94-9d28-4068-b990-a0e3775e3d47.png' width = '50%'>

$\therefore \hat{w}^* = arg \ min_{\hat{w}} \ E_{\hat{w}}$

可证这还是一个**凸函数求最值**的问题。(skipped)

Hint: 矩阵微分公式

<img src='https://user-images.githubusercontent.com/107236740/202985371-317c1a3d-ae52-4f02-8e65-c374c31282af.png' width='50%'>

$\therefore \hat{w}^* = (X^TX)^{-1} X^Ty$

# 对数几率回归 (Logistic Regression)

分类算法，以二分类为例：

**Sigmoid Function:** $f(z) = \frac{1}{1+e^{-z}}$ : $\mathbb{R} \rightarrow (0,1)$

**dmf:**

p(y=1|x) = $\frac{1}{1+e^{-(w^Tx + b)}}$ ;

p(y=0|x) = 1 - p(y=1|x)

**Then** 

p(y=1| $\hat{x}$ ; $\beta$) = $e^{{\beta}^T \hat{x}}$ / $(1+e^{{\beta}^T \hat{x}})$ = $p_1(\hat{x}; \beta)$

p(y=0 | $\hat{x}$ ; $\beta$) = 1 / $(1+e^{{\beta}^T \hat{x}})$ = $p_0(\hat{x}; \beta)$

**Therefore,** $\forall y \in (0,1)$

p(y | $\hat{x}$ ; $\beta$) = $y \cdot p_1(\hat{x}; \beta)$ + $(1-y) \cdot p_0(\hat{x}; \beta)$

**极大似然估计**

Loss Function: 

_def._ (信息熵 Entropy) $H(x) = - \ \Sigma \ p(x) \ log_{b}  p(x)$. (对不确定性的数学表达)

_def._ (相对熵；KL散度) $D_{KL}$ (p||q) = $- \ \Sigma_x \ p(x) \ log_{b}  \frac{p(x)}{q(x)} $.

_def._ (交叉熵) $- \ p(x) \ log_{b}  q(x) $

NOTE: 从机器学习三要素的 **“策略”** 而言，当我们的模拟分布与理想分布最接近时即为最优分布，因此可以通过**最小化交叉熵**这个策略来求出最优分布。

### 殊途同归

$\Sigma_{i=1}^m \ (- \ y_i \beta^T \hat{x_i} + \ ln(1 + e^{\beta^T\hat{x_i}}))$

# 二分类线性判别分析 (Bisective Classification) 

## 算法原理（模型）

![12](https://user-images.githubusercontent.com/107236740/203107382-774e4f76-5be1-4e09-a390-25052a6b0b3a.png)

## 损失函数推导（策略）

![hh](https://user-images.githubusercontent.com/107236740/203107968-78327bbc-2716-4a73-b0a9-5cae52e77336.png)

## 拉格朗日乘子法

救命，打公式好麻烦，我先搁着。

## 求解w（算法）

## 广义特征值和广义瑞利商
