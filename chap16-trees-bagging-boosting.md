# chap16-trees-bagging-boosting

## 机器学习技法

### Blending and Bagging

Motivation: validation means to select a best model from a group of good models, while aggregation means to combine a group of weak models to get a better prediction

Aggregation is kind of feature transformation due to combining the rules with different spliting criteria, and is kind of regularization due to combining extreme model and generate model

#### Math Notations

* Validation: $G(\mathbf x) = g_{t*}(\mathbf x) = argmin_{t\in[1,2,...,T]}E_{val}(g_t^-)$
* Uniform Blending: $G(\mathbf x) = sign(\sum^T_{t=1}1\cdot g_t(\mathbf x))$
* Non-uniform Blending: $G(\mathbf x) = sign(\sum^T_{t=1}\alpha_t\cdot g_t(\mathbf x))$
* Combine the predictions conditionally: $G(\mathbf x) = sign(\sum^T_{t=1}q_t(\mathbf x)\cdot g_t(\mathbf x))$, include non-uniformly: $q_t(\mathbf x) = \alpha_t$

#### Uniform Blending

In terms of classification, if the $g_t$ is very different (**diversity and democracy**), the majority is the output and can **correct minority**. The notation is 
$$
G(\mathbf x) = argmax_{1<k<K}\sum^T_{t=1}[g_t(\mathbf x) = k]
$$
In terms of Regression, if the $g_t$ is very different (**diversity and democracy**), the average could be more accurate than individual
$$
G(\mathbf x) = \frac 1 T\sum^T_{t=1}g_t(\mathbf x)
$$
Theoretically,
$$
\begin{align} avg[(g_t(\mathbf x)-f(\mathbf x))^2] & = avg(g_t^2 -2g_tf+f^2) = avg(g_t^2) - 2Gf + f^2 = avg(g_t^2) -G^2 + (G-f)^2 \\
& = avg(g^2_t - 2g_tG+G^2) + (G-f)^2 \\
& = avg((g_t-G)^2) + (G-f)^2 \\
\end{align}
$$
This means
$$
avg(E_{test}(g_t)) = avg(\epsilon(g_t-G)^2) + E_{test}(G) \ge E_{test}(G)
$$
也就是说，平均多个$g_t$的精度要高于$G$，所以uniform blending比任何单一假设好

#### Linear Blending

Optimization: $Given ~g_t, min_{\alpha_t\ge0}E_{in}(\alpha)$

For regression
$$
target:\qquad min_{\alpha_t\ge0}\frac 1N\sum^N_{n=1}(y_n-\sum^T_{t=1}\alpha_tg_t(\mathbf x_n))^2
$$
**This is the basic function expansion**

#### Ant blending (stack)

#### Bagging

Boostrapping is re-sample N examples from D **uniformly with replacement**，从D中随机采样N个样本，这些样本可以重复

Bagging means bootstrap aggregation

* for t = 1, 2, ..., T
* request size-N' data $\hat D_t$ from bootstrapping
* obtrain $g_t$ by $A(\hat D_t)$,  $G = Uniform({g_t})$ ，重复多次这个采样建模过程，然后用uniform blending

**Where the diversity comes from**

* Different models
* Different hyper-parameters
* Algorithmic randomness
* Data randomness

### Adaboost

$u^{(1)} = [\frac 1 N, ..., \frac 1 N]$

for t = 1, 2, ... , T

1. obtain $g_t$ by $A(D, u^{(t)})$ where $A$ tries to minimize $u^{(t)}-weighted~0/1~error$
2. update $u^{(t)}$ to $u^{(t+1)}$ by
   1. incorrect: $u^{(t+1)} \leftarrow u^{(t)} \cdot \Delta t $
   2. correct:  $u^{(t+1)} \leftarrow u^{(t)} / \Delta t $
   3. $\Delta t = \sqrt{1-\epsilon_t \over \epsilon_t}$
   4. $\epsilon_t = {\sum^N_{n=1}u_n^{(t)}[y_n \neq g_t(\mathbf x_n))]\over\sum^N_{n=1}u_n^{(t)}}$
3. compute $\alpha_t = ln(\Delta t)$

return $G(\mathbf x) = sign(\sum^T_{t=1}\alpha_tg_t(\mathbf x))$

**If error rate is 0, then $\Delta t = 0$**



