## Logistic Regression

### Intro

One way to build a probabilistic classifier is to **create a joint model of the form $p(y, \mathbf x)$ and then to condition on $\mathbf x$**, then **deriving $p(y|\mathbf x)$**. This is called the **generative approach**. Alternative approach is to **fit a model of the form $p(y|\mathbf x)$ directly**. This is called the **discriminative approach**

### Model specification

$$
p(y|\mathbf x, \mathbf w) = Ber(y|sigm(\mathbf w^T \mathbf x))
$$

### Model fitting

Due to the Bernoulli distribution, the negative log-likelihood for logistic regression is given by
$$
\begin{align} NLL(\mathbf w) & = -\sum_{i=1}^Nlog[\mu_i^{I(y_i=1)}\times(1-\mu_i)^{I(y_i=0)} \\ & = -\sum_{i=1}^N[y_ilog\mu_i + (1-y_i)log(1-\mu_i)]
\end{align}
$$
This is called called the **cross-entropy error** function, and this is why **exsiting `softmax_with_cross_entropy` in tensorflow**

### Gradient descent

#### Vanilla Update & Steepest Descent

```python
x += learning_rate * dx
```

**epoch**: In theory, we should sample with replacement, although in practice it is usually better on randomly premute the data and sample without replacement, and then to repeat. **A single such pass over the entire data set** is called an epoch.

#### Momentum Update

```python
v = mu * v - learning_rate * v
x += v
```

**Why momentum is needed?**

The steepest descent path with exact line searches exhibits a characteristic **zig-zag** behavior. Assume the cost function $\phi(\eta) = f(\mathbf  \theta_k + \eta\mathbf d_k)$, 

#### Nesterov Momentum

#### Learning Rate

#### Adagrad

#### RMSprop

**The improvment of RMSprop compared of Adagrad**

#### Adam

### Softmax

### Gaussian discriminat analysis

### Generative vs Discriminative classifiers 