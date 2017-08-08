# ML-Feature-Engineering

## 偏度处理

一二三四阶（中心）矩，你可以按照位置、离散程度、对称性、长尾短尾来形成初步的印象

偏度: 峰与标准正太分布相比的左右倾斜程度, 0.75, $log(1 + x + 0.000001)$
$$
\frac 1 {n-1} \sum_{i=1}^n(x-\hat x)^3/sd^3
$$
峰度: 峰的粗细/宽窄程度

## 归一化/标准化

minmax
$$
x = \frac {x - min} {max - min}
$$
z-score
$$
x = \frac {x - \mu} \sigma
$$
l2 范数
$$
x = \frac x {\sqrt {\sum_{i=1}^n x_i^2}}
$$
