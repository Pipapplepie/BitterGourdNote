# 决策树 (Decision Tree)

Recall: 信息熵、特征空间（可见Chap1-2 Note）

$H(X) = E[I(X)] = - \Sigma_x \ p(x) \log_b p(x)$ (此处以离散型为例)

约定：当p(x)=0时， $p(x) \log_b p(x) = 0$。当存在x s.t. p(x) = 1，则 H(X) = 0.

## ID3 决策树

将样本类别标记 y 视为随机变量，其样本集合为D，各类别占比为 $p_k \ \forall k = 1,2,...,|Y|$. 则D的信息熵为：

$Ent(D) = - \ \Sigma_{k=1}^{|Y|} \ p_k \ \log_2 p_k$

_def._ （条件熵：Y的信息熵关于概率分布X的期望）
$H(Y|X) = \Sigma_x \ p(x) \ H(Y|X=x)$

e.g., 从单个特征a来看，假设其取值为{ $a^1, a^2, ..., a^V$}. D为a的取值样本集合， $|D^v| / |D|$表示取值 $a^v$的样本占比。

则关于a的条件熵为： $\Sigma_{v=1}^V \frac{|D^v|}{|D|} \ Ent(D^v)$

_def._ （信息增益 Information Gain） 在已知属性a的取值后y的不确定性的减少量，即纯度的提升：

$Gain(D,a) = Ent(D) - \Sigma_{v=1}^V \frac{|D^v|}{|D|} \ Ent(D^v)$

$\rightarrow$ **ID3决策树的策略：** 以**信息增益**为准则来划分属性的决策树。

$a_* = arg \ max_{a \in A} \ Gain(D,a)$


##

##

##
