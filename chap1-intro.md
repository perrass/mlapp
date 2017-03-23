## Introduction

### Supervised Learning

#### Classification

Three types of classification: binary classification, multiclass classification, multi-label classification

The **mode** of the distribution $p(y|\mathbf x, D)$ is the value that occurs most often, and the probability of the most probable class label is
$$
\hat y = \hat f(\mathbf x) = {argmax}^C_{c=1} p(y = c | \mathbf x, D)
$$
This is also known as a **MAP estimate (maximum a posteriori)**

#### Regression

$$
y(\mathbf x) = \mathbf {w_T x} + \epsilon = \sum^D_{j=1} w_jx_j + \epsilon
$$

We often assume the $\epsilon$ has a Gaussian distribution, and $\epsilon - N(\mu, \sigma^2)$. To make the connection bewteen linear regression and Gaussians more explicit, we can rewrite the model:
$$
p(y|\mathbf x, \mathbf  {\theta}) = N(y|\mu(\mathbf x), \sigma^2(\mathbf x))
$$
where $\theta = (\mathbf w, \sigma^2)$

**To model the non-linearity relationship**, we use **basis function expansion**
$$
p(y|\mathbf x, \theta) = N(y|\mathbf w^T \phi(\mathbf x), \sigma^2)
$$
where $\phi(\mathbf x)$ is the non-linear function

E.g.

For **polynomial regression**, $\phi(\mathbf x) = [1, x, x^2,...,x^d]$

In fact, many popular machine learning methods, such as support vector machines, neural networks, classification and regression trees, can be seen as **different ways of estimating basis functions from data**

#### Logistic regression

To generalize linear regression to binary classification

1. replace the Gaussian distribution for y with a Bernoulli distribution $p(y|\mathbf x, \mathbf w) = Ber(y|\mu(\mathbf x))$
2. choose a plausible non-linear function that **ensures $0 <= \mu(\mathbf x) <= 1$** by defining, $\mu(\mathbf x) = sigm(\mathbf w^Tx)$, where $sigm(x) = {1\over{1+exp(-x)}} = {e^x\over{e^x+1}}$ 

$$
p(y|\mathbf x, \mathbf w) = Ber(y|sigm(\mathbf w^T\mathbf x))
$$

### Unsupervised Learning

The distinction between supervised and unsupervised learning

* $p(\mathbf x_i | \theta)$ compared to $p(y_i | \mathbf x_i, \theta)$
* $\mathbf x_i$ is a vector of features, so we need to create **multivariate probability models**, while, in supervised learning, $y_i$ is usually just a single variable that we try to predict

#### Clustering

Goals

1. To estimate the distribution over the number of clusters, $p(K|D)$. For simplicity, we often apporximate the distribution $p(K|D)$ by its mode, $K^* = {argmax}_K p(K|D)$ (K is the number of clusters)
2. To estimate which cluster each point belongs to.$z^*_i = {argmax}_k ~ p(z_i = k | \mathbf x_i, D)$

#### Latent factors

Latent variables are those which never observed in the training set

Dimensinality reduction, and PCA (principal components analysis) is the most common approach. This can be thought of as an unsupervised version of (multi-ouput) linear regression

#### Graph Structure

To compute $\hat G = {argmax} ~ p(G|D)$

#### Matrix completion

To infer plausible values for the missing entries. And used in, **image inpainting**, **collaborative filtering**, and **frequent itemset mining**

### Some basic concept in machine learning

**Parametric model** has a fixed number of parameters, while the number of params of **non-parametric model** grow with the amount of training data (KNN)

#### K-nearest neighbors classifier

$$
p(y = c | \mathbf x, D, K) = {1\over K} \sum_{i<- N_K(\mathbf x, D)} I(y_i = c)
$$

where $N_K(\mathbf x, D)$ are the indices of the K nearest points to $\mathbf x$ in D and $I$ is the **indicator function**

This method is an example of **memory-based learning** or **instance-based learning**

However, for high dimensional inputs, the performance of knn is poor due to the **curse of dimensinality**

#### Parametric model

The main way to combat the curse of dimensinality is to make some assumptions about the nature of the data distribution (**either $p(y|\mathbf x)$ for a supervised problem or $p(\mathbf x)$ for an unsupervised problem**). These assumptions, known as **inductive bias**, are often embodied in the form of a parametric model

### Notes

* data across a variety of domains exhibits a property known as the **long tail**
* The probabilistic approach of frequent itemset mining is more predictively accurate than association rules, but less interpretible. 

#### Model Selection

Using cross-validation