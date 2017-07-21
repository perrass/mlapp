# Chap13 Sparse linear models

## Intro

We can use **mutual information** for feature selection, but this is based on a **myopic strategy that only looks at one variable at a time**. This can fail if there are interaction effects. In this chapter, we will generalize a linear model, of the form $p(y|\mathbf x) = p(y|f(\mathbf w^T\mathbf x))$ for some link function $f$. Then we can do feature selection by making the weights **sparse**.

#### The motivation

1. For high dimensional data, especially those $D > N$, feature selection can **prevent overfitting**
2. For kernel method, the resulting design matrix has size $N\times N$. Feature selection in this context is equivalent to **selecting a subset of the training examples**, which can help reduce overfitting and **computational cost**. 

## Bayesian variable selection

Suppose there exists $D$ features, then the combinations for features are $2^D$, which is impossible to compute the full posterior in general. Hence, the main topic of variable selection is **algorithmic speedups**

### Greedy

The basic ideas behind greedy algorithm are:

1. Computing argmax $p(D|\mathbf w)$
2. Evaluating its marginal likelihood $\int p(D|\mathbf w)p(\mathbf w)d\mathbf w$

Forward and backward stepwise logistic regression is using this idea

## L1 regularization: basics

#### The Laplace Distribution

$$
Lap(x|\mu, b) = \frac 1 {2b} exp(-\frac {|x-\mu|} {b})
$$

Properties:

* The mean = $\mu$, mode = $\mu$, var = $2b^2$
* It is robust to outliers
* Put more probability density at 0 than the Gaussian. This leads to encourage sparsity in a model

![](assets/laplace_outliers.png)

#### Definition

In general, the technique of **putting a zero-mean Laplace prior on the parameters and performing MAP estimation is called $L1$ regularization**. In detail, the form of prior is 
$$
p(\mathbf w|\lambda) = \prod^D_{j=1}Lap(w_j|0, 1/\lambda)\propto \prod^D_{j=1}e^{-\lambda |w_j|}
$$
Using the negative log likelihood to perform MAP estimation with this prior:
$$
f(\mathbf w) = -logp(D|\mathbf w)-logp(\mathbf w|\lambda) = NLL(\mathbf w) + \lambda ||\mathbf w||_1
$$
This can be thought of as a **convex approximation to the non-convex l_0 objective**

In the case of linear regression, the $l_1$ objective becomes
$$
f(\mathbf w) = \sum^N_{i=1} -\frac 1 {2\sigma^2}(y_i-(w_0+\mathbf w^T\mathbf x_i))^2 + \lambda ||\mathbf w||_1 = RSS(\mathbf w) + \lambda' ||\mathbf w||_1
$$
where $\lambda' = 2\lambda \sigma^2$

### Why does l1 yield sparse solution

Changing the non-smooth objective function to a smooth objective function with constrains
$$
min_{\mathbf w} RSS(\mathbf w) \qquad s.t. ||\mathbf w||_1 \le B
$$
This equation is known as **lasso**, which stands for "least absolute shrinkage and selection operator"

To solve this, we set **subgradient** of a (convex) function $f: \tau -> \Bbb R$ at a point $\theta_0$ to be a scalar $g$ such that
$$
f'(\theta) - f(\theta_0) \ge g(\theta - \theta_0) \forall\theta \in \tau
$$
where $\tau$ is some interval containing $\theta_0$. We define the **set of subderivatives** as the **interval $[a, b]$ where $a$ and $b$ are the one-sided limits**
$$
a = lim_{\theta \to \theta_0^-}\frac {f(\theta ) - f(\theta_0)} {\theta - \theta_0}, \quad b = lim_{\theta \to \theta_0^+}\frac {f(\theta ) - f(\theta_0)} {\theta - \theta_0} 
$$
The set $[a, b]$ of all subderivatives is called the **subdifferential** of the function $f$ at $\theta_0$ and is denoted $\partial f(\theta)|_{\theta_0}$. For example, in the case of the absolute value function $f(\theta) = |\theta|$, the subderivative is given by
$$
\partial f(\theta) = \begin{cases}
& \{-1\}, &\quad if ~\theta < 0\\
& [-1, 1], &\quad if ~\theta = 0\\
& \{+1\}, &\quad if ~\theta > 0
\end{cases}
$$
For lasso, we have
$$
\hat w_j(c_j) = \begin{cases}
& (c_j +\lambda) / a_j & \quad if ~ c_j <\lambda\\
& 0 & \quad if ~ c_j \in [-\lambda, \lambda]\\
& (c_j -\lambda) / a_j & \quad if ~ c_j >\lambda\\
\end{cases}
$$
where $a_j = 2\sum^n_{i=1}x^2_{ij}$ and $c_j = 2\sum^n_{i=1}x_{ij}(y_i-\mathbf w^T_{-j}\mathbf x_{i, -j})$. Hence, there exists sparsity.

### Model selection

A downside of using $l_1$ regularization to select variables is that it can give quite different results if the data is perturbed sightly. A frequentist solution to this is to use **bootstrap sampling**, and to rerun the estimator on different versions of the data. This method is known as **stability selection** (PS: it should be faster than random forest). We can threshold the stability selelction probabilities at some level, say 90%, and thus derive a sparse estimator. This is known as **bootstrap lasso** or **bolasso**

## L1 regularization: algorithms

### Coordinate descent

### LARS

### Proximal and gradient projection methods

## L1 regularization: extensions

### Group lasso

### Fused lasso

## Automatic relevance determination / sparse Bayesian learning