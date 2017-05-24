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

#### Classification cost

In the classification setting, there are several ways to measure the quality of a split. First, we fit a multinouli model to the data in the leaf satisfying the test $X_j < t$ by estimating the class-conditional probabilities as follows:
$$
\hat \pi_c = {1\over{|D|}}\sum_{i\in D}I(y_i = c)
$$
where D is the data in the leaf. Given this, there are several common error measures for evaluating a proposed partition.

##### Misclassification rate. 

We define the most probable class label as $\hat y_c = argmax_c \hat \pi_c$. The corresponding error rate is $1-\hat \pi_c$

##### Entropy or deviance

$$
H(\hat \pi_c) = -\sum^C_{c=1}\hat \pi_clog\hat \pi_c
$$

Minimizing the entropy is equivalent ot maximizing the **information gain** between test $X_j < t$ and the class label $Y$
$$
infoGain(X_j < t, Y) = H(Y) - H(Y|X_j < t) = (-\sum_cp(y=c)log~p(y=c)) + (\sum_c p(y=c|X_j < t)log ~p (c|X_j < t))
$$
If $X_j$ is categorical, and we use tests of the form $X_j = k$, then **taking expectations over values of $X_j$ gives the mutual information between $X_j$ and Y**
$$
E(infoGain(X_j,Y)) = \sum_k p(X_j=k)infoGain(X_j = k, Y) = H(Y) - H(Y|X_j) = I(Y; X_j)
$$

##### Gini index

Since $\hat \pi_c$ is an MLE for the distribution $p(c|X_j < t)$
$$
\sum^C_{c=1}\hat \pi_c(1-\hat \pi_c)=\sum_c\hat \pi_c - \sum_c\hat \pi_c^2 = 1-\sum_c\hat \pi_c^2
$$
In the two-class case, the msiclassification rate is $1-max(p, 1-p)$, the entropy is $H(p)$, and the Gini index is $2p(1-p)$

#### Pros and Cons of trees

Pros

* Easy to interpret
* Easily handle mixed discrete and continuous inputs
* **insensitive to monotone transformations of the input**(since the split points are based on ranking the data points)
* automatic variable selection
* relatively robust to outliers
* scale well to large data sets
* can be modified to handle missing inputs

Cons

* Do not predict very accurately compared to other models. (This is in part due to the **greedy nature of the tree construction algorithm**
* **Unstable**, small changes to the input data can have large effects on the structure of the tree (High variance in frequentist terminology)

#### Random forest

One way to reduce the variance of an estimate is to average together many estimates. For example, we can train M different trees on different subsets of the data, chosen randomly with replacement, and then compute the ensemble
$$
f(\mathbf x) = \sum^M_{m=1}\frac 1 M f_m(\mathbf X)
$$
where $f_m$ is the m'th tree. This technique is called **bagging**, which stands for **bootstrap aggregating**

Random forest goes further by **learning trees based on a randomly chosen subset of input variables, as well as a randomly chosen subset of data cases to decorrelate the base learner**

