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

$a_* = arg \ max_{a \in A} \ Gain(D,a)$ (选择最优划分属性)


## C4.5决策树

ID3的bug：对可能**取值数目较多的属性**有所偏好。（如无意义的编号属性）

**改进：** 增益率

$Gain \ ratio(D,a) = \frac{Gain(D,a)}{IV(a)}$ where

$IV(a) = - \Sigma_{v=1}^V \frac{|D^v|}{|D|} \ \log_2 \frac{|D^v|}{|D|}$ （a的固有值）

a 的可能取值个数|V|越大，通常其固有值IV(a)也越大。**但**增益率可能又会对可能取值数目**较少**的属性有偏好。

**继续改进：** 折中

先选出 **信息增益** 高于平均水平的属性，然后从中选择 **增益率** 最高的。

## CART决策树

_def._ （基尼值） 从样本集合D中随机抽取两个样本，其类别标记不一致的概率。

$Gini(D) =\Sigma_{k=1}^{|Y|} \Sigma_{k' \neq k} p_k p_{k'}$

$= 1 -\Sigma_{k=1}^{|Y|} p_k^2$

_def._ （基尼指数） （类似于信息熵和条件熵的关系）

Gini_index(D,a) = $\Sigma_{v=1}^V \frac{|D^v|}{|D|} \ Gini(D^v)$

**CART的策略**：选择基尼指数最小的属性为最优划分属性

$a_* = arg \ max_{a \in A} \  Gini \ index(D,a)$

最优划分点的选择：对属性a的每个可能值v，将数据集D分为a=v和$a \neq v$两部分来计算基尼指数，即

Gini_index(D,a) = $\frac{|D^{a=v}|}{|D|} \ Gini(D^{a=v}) + \frac{|D^{a \neq v}|}{|D|} \ Gini(D^{a \neq v})$

##
