# Mixture models and the EM algorithm

## Latent variable models

Graphical models can be used to define high-dimensional joint probability distributions. The basic idea is to model **dependence** between two variables by adding an edge between them in the graph.

An alternative approach is to assume that **the observed variables are correlated because they arise from a hidden common "cause"**. Advantage:

1. Latent variable models often have fewer parameters than directly represent correlation in the visible space (**but, now deep neural network can overdraw this**)
2. The hidden variables in an LVM can serve as a **bottleneck**, which computes a compressed representation of the data

## Mixture models

The simplest form of LVM is when $z_i \in \{1, ..., K\}$, representing a discrete latent state. The discrete prior is $p(z_i) = Cat(\pi)$, the likelihood is $p(\mathbf x_i|z_i=k) = p_k(\mathbf x_i)$. Then the **mixture model** is 
$$
p(\mathbf x_i |\theta) = \sum^K_{k=1}\pi_k p_k(\mathbf x_i|\theta)
$$

| Likelihood      | Prior           | Name                              |
| --------------- | --------------- | --------------------------------- |
| MVN             | Discrete        | Mixture of Gaussians              |
| Prod. Discrete  | Discrete        | Mixture of multinomials           |
| Prod. Gaussian  | Prod. Gaussian  | Factor analysis/probabilistic PCA |
| Prod. Gaussian  | Prod. Laplace   | Probabilistic ICA / sparse coding |
| Prod. Discrete  | Prod. Gaussian  | Multinomial PCA                   |
| Prod. Discrete  | Dirichlet       | Latent Dirichlet allocation       |
| Prod. Noisy-OR  | Prod. Bernoulli | BN20/QMR                          |
| Prod. Bernoulli | Prod. Bernoulli | Sigmoid belief net                |

where Prod. means product, i.e., Prod. Gaussian means $\prod_j N(x_{ij}|\mathbf z_i)$

There are two main applcations of mixture models:

1. As a black-box density model $p(\mathbf x_i)$. This can be useful for a variety of tasks, such as data compression, outlier detection, and creating generative classifiers
2. Clustering

### Clustering

We first **fit the mixture model**, and then compute $p(z_i=k|\mathbf x_i, \theta)$, which represents the **posterior probability** that point $i$ belongs to cluster $k$. This is known as the **responsibility** of cluster $k$ for point $i$, and can be computed using Bayes rule
$$
r_{ik} = p(z_i=k|\mathbf x_i, \theta) = \frac {p(z_i = k |\theta)p(\mathbf x_i| z_i = k, \theta)} {\sum^K_{k'= 1}p(z_i = k' |\theta)p(\mathbf x_i| z_i = k', \theta)}
$$
This procedure is called **soft clustering**, and is identical to the computations performed when using a generative classifier. The difference between the two models only arises at training time, in the mixture case, we never observe $z_i$, whereas with a generative classifier, we do observe $y_i$

Then using **MAP estimate**, we can compute a **hard clustering**
$$
z_i^* = argmax_k ~r_{ik} = argmax_k ~logp(\mathbf x_i | z_i = k, \theta) + log p(\mathbf z_i = k|\theta)
$$

## The EM algorithm

### Basic idea

Let $\mathbf x_i$ be the visible or observed variables in case $i$, and let $\mathbf z_i$ be the hidden or missing vairables. The goal is to **maximize the log likelihood of the observed data**
$$
l(\theta) = \sum^N_{i=1}logp(\mathbf x_i|\theta) = \sum^N_{i=1} log[\sum_{\mathbf z_i} p(\mathbf x_i, \mathbf z_i |\theta)]
$$
Define the **complete data log likelihood** to be 
$$
l_c(\theta) = \sum^N_{i=1} logp(\mathbf x_i, \mathbf z_i |\theta)
$$
The problem is $\mathbf z_i$ is unknown. So let us define the **expected complete data log likelihood**
$$
Q(\theta, \theta^{t-1}) = E[l_c(\theta) | D, \theta^{t-1}]
$$
where $t$ is the current iteration number. $Q$ is called the **auxiliary function**. The expectation is taken wrt the old parameters, $\theta^{t-1}$, and the observed data $D$. The goal of the **E step** is to compute $Q(\theta, \theta^{t-1})$. Then, the **M step**, we optimize the Q function wrt $\theta$, $\theta^t = argmax_{\theta} Q(\theta, \theta^{t-1})$

The perform MAP estimation, we modify the M step as follows.
$$
\theta^t = argmax_{\theta} Q(\theta, \theta^{t-1}) + logp(\theta)
$$
**The EM algorithm monotonically increases the log likelihood of the observed data, or it stays the same**. 

### EM for GMMs

#### Auxiliary function

The expected complete data log likelihood is given by
$$
\begin{align}
Q(\theta, \theta^{t-1}) & = E[\sum_i logp(\mathbf x_i, z_i | \theta)] \\ 
& = \sum_i E[log[\prod^K_{k=1}(\pi_kp(\mathbf x_i|\theta_k))^{I(z_i=k)}]] \\
& = \sum_i \sum_k E[I(z_i=k)]log(\pi_k p(\mathbf x_i|\theta_k)) \\
& = \sum_i\sum_k p(z_i=k|\mathbf x_i, \theta^{t-1})log(\pi_kp(\mathbf x_i|\theta_k))\\
& = \sum_i\sum_k r_{ik}log_{\pi_k} + \sum_i\sum_k r_{ik}log p(\mathbf x_i |\theta_k)
\end{align}
$$
where $r_{ik} = p(z_i = k|\mathbf x_i, \theta^{t-1})$ is the **responsibilty** that cluster $k$ takes for data point $i$. **This is computed in the E step**

#### E step

The E step is **the same for any mixture model** and is **to compute the responsibility**
$$
r_{ik} = \frac {\pi_k p(\mathbf x_i |\theta^{t-1}_k)} {\sum_{k'}\pi_{k'}p(\mathbf x_i | \theta_{k'}^{t-1})}
$$

#### M step

In the M step, we **optimize $Q$ wrt $\pi$ and the $\theta_k$**. For $\pi$,
$$
\pi_k = \frac 1 N \sum_i r_{ik} = \frac {r_k} {N}
$$


### Theoretical basis

## Model selection for latent variable mdoels



