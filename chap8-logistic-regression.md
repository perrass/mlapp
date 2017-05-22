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

Develop a stable method for **picking up the step size**, so the method is guaranteed to converge to a **local optimum** no matter where we start. (This property is called **global convergence**, which should not be confused with convergence to the global optimum). 

Assume the cost function $\phi(\eta) = f(\mathbf  \theta_k + \eta\mathbf d_k) \approx f(\theta) + \eta\mathbf g^T \mathbf d$, where $\mathbf d$ is our descent direction. To optimize the function, we have $\eta_k = argmin_{\eta > 0} \phi(\eta)$. A neccessary condition for the optimum is $\phi'(\eta) = 0$. By the chain rule, $\phi'(\eta) = \mathbf d^T \mathbf g$, where $\mathbf g = f'(\theta + \eta \mathbf d)$ is the gradient at the end of the step. So **we either have $\mathbf g = \mathbf 0$, which means we have found a stationary point, or $\mathbf g \bot \mathbf d$, which means that exact search stops at a point where the local gradient is perpendicular to the search direction**. Hence consecutive directions will be orthogonal.

One simple heuristic to reduce the effect of  **zig-zag** behavior is to add a momentum term. 

PS: So **the direct of gradient decent is not orthogonal to the former step, due to the learning rate is not optimal**.

#### Nesterov Momentum

```python
x_head = x + mu * v
v = mu * v - learning_rate * dx_head
x += v
```

#### Learning Rate

Learning rate with fixed size would be too high when the gradient is too small. 

1. Step decay, reducing the learning rate by a half every 5 epochs or by 0.1 every 20 epochs
2. exponential decay $\alpha = \alpha_0e^{-kt}$, $\alpha, k$ are hyper parameters, and t is the iterations
3. $1 / t$ decay, $\alpha = {\alpha_0 \over {1+kt}}$ 

#### Adagrad

```python
cache += dx ** 2
x += -learning_rate * dx / (np.sqrt(cache) + eps)
```

eps is the smooth parameter to ensure the denominator is not zero, which value is 1e-6 to 1e-8

#### RMSprop

```python
cache = decay_rate + (1 - decay_rate) * dx ** 2
x += -learning_rate * dx / (np.sqrt(cache) + eps)  # the range of decay_rate is [0.9, 0.99, 0.999] 
```

**The improvment of RMSprop compared of Adagrad**

The new method to calculate cache ensure the update of cache is **monotonically smaller**

#### Adam

```python
m = beta1 * m + (1 - beta1) * dx		# the default value of beta1 is 0.9
v = beta2 * v + (1 - beta2) * dx ** 2   # the default value of beta2 is 0.999
x += -learning_rate * m / (np.sqrt(v) + eps)
```

**adam is the default optimizer for many cases, and SGD + nesterov momentum is an alternative**

### Softmax

### Gaussian discriminat analysis

One

### Generative vs Discriminative classifiers 

