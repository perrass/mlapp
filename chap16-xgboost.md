# Chap16 Xgboost

### Intro

For supervised learning, the basic idea is to optimize objective function to get the best parameters given the training data. The objective functions is **must always** contain two parts, training loss and regularization. The regularization term **controls the complexity of the model**
$$
Obj(\Theta) = L(\theta) + \Omega(\Theta)
$$
MSE for Regression
$$
L(\theta) = \sum_i(y_i-\hat y_i)^2
$$
Logistic loss for logistic regression
$$
L(\theta) = \sum_i[y_iln(1+e^{-\hat y_i}) +(1-y_i)ln(1+e^{\hat y_i})]
$$

### Tree Ensemble

![](/assets/Tree_Ensemble.png)

The prediction scores of each individual tree are summed up to get the final score. Mathematically, we can write our model in the form
$$
\hat y_i = \sum^K_{k=1}f_k(x_i), f_k \in F
$$
where $K$ is the number of trees, $f$ is a function in the functional space $F$, and $F$ is the set of all possible CARTs. Therefore our objective to optimize can be written as
$$
obj(\theta) = \sum^n_il(y_i, \hat y_i) + \sum^K_{k=1}\Omega(f_k)
$$

> Now here comes the question, what is the model for random forests? It is exactly tree ensembles. So **random forests and boosted trees are not different in terms of model, the difference is how we train them**. This means if you write a predictive service of tree ensembles, you only need to write one oof them and they should directly work for both.

### Tree Boosting

#### Additive Training

To get the **parameters of trees, we use an additive strategy**. You can find that what we need to learn are those function $f_i$, with each **containing the structure of the tree and the leaf scores**
$$
\begin{align} \hat y^0_i & = 0 \\
\hat y^1_i & = f_1(x_i) = \hat y^0_i +f_1(x_i) \\
\hat y^2_i & = f_1(x_i) + f_2(x_i) = \hat y^1_i + f_2(x_i) \\ 
...\\
\hat y^t_i & = \sum^t_{k=1}f_k(x_i) = \hat y^{t-1}_i + f_t(x_i)
\end{align}
$$
If we use MSE as our loss function, the added one tree is to optimize our objective
$$
\begin{align} obj^t & = \sum^n_{i=1}l(y_i, \hat y_i^t) + \sum^t_{i=1}\Omega(f_i) \\
	& = \sum^n_{i=1}l(y_i, \hat y_i^{t-1} + f_t(x_i)) + \Omega(f_t) + constant
\end{align}
$$

$$
\begin{align} obj^t & = \sum^n_{i=1} (y_i-(\hat y_i^{t-1}+f_t(x_i)))^2 + \sum^t_{i=1}\Omega(f_i)\\
& = \sum^n_{i=1}[2(\hat y^{t-1}_i - y_i)f_t(x_i) + f_t(x_i)^2] + \Omega(f_t) + constant

\end{align}
$$

where $constant ~ is \sum^n_{i=1}(y_i - \hat y^{t-1}_i)^2$

The form of MSE is friendly, with a first order term (**residual**) and a quadtratic term. For other losses of interest (e.g. logistic loss), it is not so easy to get such a nice form.

So in general case, we take the Taylor expansion of the loss function up to the second order.
$$
obj^t = \sum^n_{i=1}[l(y_i, \hat y^{t-1}_i) + g_if_t(x_i) + \frac 1 2h_if^2_t(x_i)] + \Omega(f_t) + constant
$$
where
$$
\begin{align} g_i & = \partial_{\hat y_i^{t-1}}l(y_i, \hat y_i^{t-1}) \\
h_i & = \partial^2_{\hat y_i^{t-1}}l(y_i, \hat y_i^{t-1})
\end{align}
$$
After we remove all the constants, the specific objective at step t becomes
$$
\sum^n_{i=1}[g_if_t(x_i) + \frac 1 2h_if_t^2(x_i)] + \Omega(f_t)
$$
This becomes our **optimization goal for the new tree**. One important advantage of this definition is that it only depends on $g_i$ and $h_i$. This is how xgboost can **support custom loss functions**.

#### Regularization term

Redefine the definition of the tree $f(x)$ as 
$$
f_t(x) = w_{q(x)}, w \in R^T, q: R^d \to [1, 2, ..., T]
$$
Here $w$ is the vector of scores on leaves, $q$ is a function assigning each data point to the corresponding leaf, and T is the number of leaves. In XGBoost, the complexity is 
$$
\Omega(f) = \gamma T + \frac 1 2 \lambda \sum^T_{j=1}w^2_j
$$

> The regularization is one part most tree packages treat less carefully, or simply ignore. This was because the **traditional treatment of tree learning only emphasized improving impurity, while the complexity control was left to heuristics**

#### The Structure Score

After reformalizing the tree model, we can write the objective value with the $t-th$ tree as:
$$
\begin{align} Obj^t & \approx \sum^n_{i=1}[g_iw_{q(x_i)} + \frac 1 2 h_i w^2_{q(x_i)}] + \gamma T + \frac 1 2 \lambda \sum^T_{j=1}w^2_j \\
	& = \sum^T_{j=1}[(\sum_{i\in I _j}g_i)w_j  + \frac 1 2(\sum_{i\in I_j}h_i + \lambda)w^2_j] + \gamma T
\end{align}
$$
where $I_j = \{i|q(x_i) = j\}$ is the set of indices of data points assigned to the j-th leaf. Then,
$$
obj^t = \sum^T_{j=1}[G_jw_j + \frac 1 2 (H_j + \lambda)w^2_j] + \gamma T
$$
where $G_j = \sum_{i\in I_j}g_i$ and $H_j = \sum_{i\in I_j}h_i$. In this equation $w_j$ are independent with respect to each other, the form $G_jw_J + \frac 1 2 (H_j+\lambda)w^2_j$ is quadratic and the best $w_j$ for a given structure $q(x)$ and the best objective reduction we can get is 
$$
w^*_j = - \frac {G_j} {H_j + \lambda}\\
obj^* = -\frac 1 2\sum^T_{j=1}\frac {G^2_j}{H_j+\lambda} + \gamma T
$$
![](/assets/XGBoost_obj.png) 

#### Learn the Tree Structure

To avoid get all possible tree, we should try to optimize one level of the tree at a time. In XGBoost, the gain is
$$
Gain = \frac 1 2 [\frac {G^2_L} {H_L + \lambda} + \frac {G^2_R} {H_R + \lambda} + \frac {(G_L + G_R)^2} {H_L + H_R + \lambda}] - \gamma
$$
The formula can be decomposed as:

1. the score on the new left leaf
2. the score on the new right leaf
3. the score on the original leaf
4. regularization on the additional leaf

**Pre-pruning**

> If the gain is smaller than $\gamma$, we would do better not to add that branch

**Realistic consideration**

For real valued data, we usually want to search for an optimal split. To efficiently do so, we  **place all the instances in sorted order**. A left to right scan is sufficient to calculate the structure score of all possible split solutions, and we can find the best split efficiently

[Introduction to Boosted Trees](https://xgboost.readthedocs.io/en/latest/model.html)