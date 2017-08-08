# ML-Data-Preprocessing

## 异常值检测

#### 统计

$F_L, F_U$ are the 25% and 75% respectively

$D = F_U - F_L$

小于$F_L-mD$或者大于$F_U+mD$的点是异常值, m一般为1.5

#### 业务

比如年龄，杠杆比例过高

## 缺失值填补

均值，0，回归