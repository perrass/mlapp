# Adaptive basis function models

### Intro

In chap14 and chap15, we discussed kernel methods, which provide a powerful way to create non-linear models for regression and classification. The prediction takes the form $f(\mathbf x) = \mathbf w^T\phi(\mathbf x)$, where we define$\phi(\mathbf x) =[\kappa(\mathbf x, \mu_1),...,\kappa(\mathbf x, \mu_N)] $

An alternative approach is to dispense with kernels altogether, and try to learn useful features $\phi(\mathbf x)$ directly from the input data. That is, we will create what we call an **adaptive basis-function model**, which is a model of the form
$$
f(\mathbf x) = w_0 + \sum^M_{m=1}w_m\phi_m(\mathbf x)
$$
This framework covers all of the models we will discuss this chapter

### Classification and regression tree

#### Basics

Finding a optimal partitioning of the data is NP-complete, so it is common to use the **greedy procedure**, this method is used by CART, C4.5, ID3 which are three popular implementations of the method.

The split fucntion chooses the **best feature**, and the **best value** for that feature, as follows:
$$
(j^*, t^*) = arg ~min_{j\in[1,...,D]} ~ min_{t \in T_j}~[cost({\mathbf x_i, y_i:x_{ij} \le t}) + cost(\mathbf x_i, y_i: x_{ij} > t)]
$$
The set of possible threshold $T_j$ for feature j can be obtained by sorting the unique values of $x_{ij}$. In case of categorical features, the most common approach is to consider splits of the form $x_{ij}=c_k$ and $x_{ij} \neq c_k$. **Although we chould allow for multi-way splits, this would result in data framentation, meaning too little data might fall into each subtree, resulting in overfitting**

#### Stopping heuristics

* Is the reduction in cost too small?
* Has the tree exceeded the maximum desired depth?
* Is the distribution of the response in either $D_L$ or $D_R$ pure?
* Is the number of examples in either $D_L$ or $D_R$ too small

#### Regression cost

In the regression setting, we define the cost as follows:
$$
cost(D) = \sum_{i \in D} (y_i - \hat y) ^2
$$
If we use average, $\hat y = \overline y = \sum_{i\in D}y_i / D $, alternatively, we can fit a linear regression model for each leaf, using as inputs the features that were chosen on the path from the root, and then measure the residual error 