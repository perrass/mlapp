# ML-Feature-Selection

## 相关性

数值，pearson correlation
$$
p_{x,y} = \frac {cov(x,y)} {\sigma_x\sigma_y}
$$
离散，卡方检验 chi-square

## 信息论

互信息
$$
H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y) \to H(X) - H(X|Y) = H(Y) - H(Y|X)\\
$$
于是
$$
\begin{align}
I(X, Y) & = H(Y) - H(Y|X) \\
& = \sum_x p(x)(\sum_yp(y|x)logp(y|x)) - \sum_ylogp(y)(\sum_x p(x,y)) \\
& = \sum_{x,y}p(x)p(y|x)logp(y|x) - \sum_{x,y}p(x,y)logp(y) \\
& = \sum_{x, y}p(x,y)log\frac {p(x,y)}{p(x)} -\sum_{x,y}p(x,y)log(y)\\
&=\sum_{x,y}p(x,y)log\frac {p(x,y)} {p(x)p(y)}
\end{align}
$$

## Lasso



## Tree-based

计算特征A的划分增益，每颗树都可以得到一个值，然后取个平均，这就是基于GINI系数的计算

## Step-wise