# Chap9 Generalized Linear Models and the Exponential Family

## The Exponenital Family

1. Under certain regularity conditions, the exponential family is **the only family of distributions with finite-sized sufficient statistics**, meaning that we can compress the data into a fixed-sized summary without loss of information
2. **The only family of distributions for which conjugate priors exist**, which simplifies the computation of the posterior
3. The family of distributions that makes the **least set of assumptions subject to some user-chosen constraints**
4. At the core of generalized linear models
5. At the core of variational inference

### Sufficiency

> Let $X_1$, ..., $X_n$ be a random sample from a probability distribution with unknown parameter $\theta$. Then, the statistic $Y = u(X_1, ..., X_n)$ is said to be **sufficient** for $\theta$ if the conditional distribution of $X_1, ..., X_n$ given the statistic Y, does not depend on the parameter $\theta$

Example: 

Consider a random sample $X_1, ..., X_n$ of size n, from the Bernouli distribution with $P(x = 0) = 1- p$. The parameter is the population  proportion p. **Show that $T = \sum^n_{i=1}X_i$ is a sufficient statistic**

Solution:

By independence, the joint distribution of the random sample is 
$$
\prod^n_{i=1}p^{x_i}(1-p)^{1-x_i} = p^{\sum^n_{i=1}x_i}(1-p)^{n-\sum^n_{i=1}x_i}
$$
and then the joint probability of $X_1= x_1, ..., X_n = x_n$ and $T = \sum^n_{i=1}X_i=t$ is
$$
f(x_1, x_2, ..., x_n, t; p) = p^t(1-p)^{(n-t)} \qquad where \sum^n_{i=1} x_i = t
$$
and is zero elsewhere. Further, $T = \sum^n_{i=1}X_i$ has the binomial distribution
$$
g(t; p) = \binom {n} {t}p^t(1-p)^{n-t} \qquad for ~t = 0, 1, ..., n
$$
Consequently, the conditional distribution of the sample, given $T=t$ is 
$$
f(x_1, x_2, ..., x_n|T=t, p) = \frac {f(x_1, x_2, ..., x_n, t; p)} {g(t; p)} = \frac {p^t(1-p)^{(n-t)} } {\binom {n} {t}p^t(1-p)^{n-t}} = \frac {1} {\binom n t}
$$
which **does not depend on the parameter $p$**. Hence,   $T = \sum^n_{i=1}X_i$ is a sufficient statistic.

**PS: 性质1的意思是说，指数族的一些统计量，不会受分布参数的影响，但会受分布的影响**

### Definition

A pdf or pmf $p(\mathbf x | \theta)$, for $\mathbf x = (x_1, ..., x_m) \in\chi^m$ and $\theta \in \Theta \subseteq \Bbb R^d$, is said to be in the **exponential family** if it is the form 
$$
\begin{align} P(\mathbf x | \theta) & = \frac {1} {Z(\theta)}h(\mathbf x)exp[\theta^T\phi(\mathbf x)] \\
& = h(\mathbf x)exp[\theta^T\phi(\mathbf x)-A(\theta)]
\end{align}
$$
where
$$
\begin{align}
Z(\theta) & = \int h(\mathbf x)exp[\theta^T\phi(\mathbf x)]d\mathbf x \\
A(\theta) & = logZ(\theta)
\end{align}
$$

* $\theta$ are called the **natural parameters** or **canonical parameters**
* $\phi(\mathbf x) \in \Bbb R^d$ is called a vector of **sufficient statistics**
* $Z(\theta)$ is called the **partition function**
* $A(\theta)$ is called the **log partition function** or **cumulant function**
* $h(\mathbf x)$ is the a **scaling constant**, often 1

## Generalized Linear Models

## Prohit Regression

