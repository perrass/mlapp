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

### Why does l1 yield sparse solution

**subgradient**

### Comparison of least squares, lasso, ridge and subset selection

## L1 regularization: algorithms

### Coordinate descent

### LARS

### Proximal and gradient projection methods

## L1 regularization: extensions

### Group lasso

### Fused lasso

## Automatic relevance determination / sparse Bayesian learning