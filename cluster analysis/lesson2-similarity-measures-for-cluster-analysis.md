# Lesson2: Similarity Measures for Cluster Analysis

A good clustering method will produce high quality clusters which should have **high intra-class similarity** and **low inter-class similarity**

### Numerical Features

**Minkowski distance**: 
$$
d(i, j) = \sqrt[4]{{|x_{x1}-x_{j1}|}^p + {|x_{x2}-x_{j2}|}^p+...+{|x_{xl}-x_{jl}|}^p}
$$

### Bineary Attributes

|      | 1     | 0     | sum   |
| ---- | ----- | ----- | ----- |
| 1    | q     | r     | q + r |
| 0    | s     | t     | s + t |
| sum  | q + s | r + t | p     |

distance for symmetric binary variables (the prob of each case is 50%):
$$
d(i, j) = {r+s\over{q + r + s + t}}
$$
distance for asymmetric binary variables:
$$
d(i, j) = {r+s\over{q + r + s}}
$$

| Name | Gender | Fever | Cough | Test-1 | Test-2 | Test-3 | Test-4 |
| ---- | ------ | ----- | ----- | ------ | ------ | ------ | ------ |
| Jack | M      | Y     | N     | P      | N      | N      | N      |
| Mary | F      | Y     | N     | P      | N      | P      | N      |
| Jim  | M      | Y     | P     | N      | N      | N      | N      |

* **Gender is a symmetric attribute** (not counted in) and the remaining attribtues are asymmetric binary
* d(jack, mary) = 0.33; d(jack, jim) = 0.67; d(jim, mary) = 0.75

### Categoical data

* Using one-hot encoding to generate a large number of binary attributes
* For ordinal variables, using a 0-1 scaler, e.g freshmean:0, sophomore: 1/3, junior: 2/3, senior: 1

### Mixed Type

$$
d(i, j) = {\sum^p_{f=1}w^{f}_{ij}d^f_{ij}\over{\sum^p_{f=1}w^{f}_{ij}}}
$$

A question is **how to define $w_{ij}$**

### Cosine similarity

