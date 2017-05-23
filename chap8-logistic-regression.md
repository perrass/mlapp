## Logistic Regression

### Intro

One way to build a probabilistic classifier is to **create a joint model of the form $p(y, \mathbf x)$ and then to condition on $\mathbf x$**, then **to derive p(y|\mathbf x)$ by Bayes rules**. This is called the **generative approach**. Alternative approach is to **fit a model of the form $p(y|\mathbf x)$ directly**. This is called the **discriminative approach**

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

One important application of MVNs is to define the class conditional densities in a generative classifier 
$$
p(\mathbf x | y = c, \mathbf \theta) = N(\mathbf x | \mathbf \mu_c, \mathbf \Sigma_c)
$$
The resulting technique is called Gaussian discriminat analysis. If $\mathbf \Sigma_c$ is **diagnoal**, this is equivalent to **naive Bayes**

Using Bayes rules, we can classify a feature vector
$$
\hat y(\mathbf x) = argmax_x[log~p(y=c|\pi) + log~p(\mathbf x | \mathbf \theta_c)]
$$
When we compute the probability of $\mathbf x$ under each class conditional density, we are measuring the distance from $\mathbf x$ to the center of each class, $\mathbf \mu_c$, using **Mahalanbis distance**. This can be thought of as a **nearest centroids classifier**

#### LDA (Linear discriminant analysis) 

When considering a special case in which the **convariance matrices are tied or shared across classes, $\mathbf \Sigma_c = \Sigma$
$$
p(y = c | \mathbf x, \mathbf \theta) \sim exp[\mu^T_c\Sigma^{-1}\mathbf x - \frac 1 2 \mu^T_c\Sigma^{-1}\mathbf \mu_c + log ~\pi_c] exp[-\frac 1 2 \mathbf x\Sigma^{-1}\mathbf x]
$$
Define
$$
\gamma_c = - \frac 1 2 \mu^T_c\Sigma^{-1}\mathbf \mu_c + log ~\pi_c \\
\beta_c = \Sigma^{-1}\mu_c
$$
then
$$
p(y = c | \mathbf x, \mathbf \theta) = {{e^{\beta_c^T + \gamma_c}} / \sum_{c'}e^{\beta_{c'}^T + \gamma_{c'}}} = \mathbf S(\eta)_c
$$
where S is the **softmax function**

#### Two classes LDA

The posterior is given by
$$
p(y = 1 | \mathbf x, \mathbf \theta) = sigmoid((\beta_1 - \beta_0)^T\mathbf x + (\gamma_1 - \gamma_0))
$$
if we define
$$
\mathbf w = \beta_1 - \beta_0 = \Sigma^{-1}(\mu_1 - \mu_0) \\
\mathbf x_0 = \frac 12 (\mu_1 + \mu_0) - (\mu_1 - \mu_0){log(\pi_1/\pi_2)\over{(\mu_1-\mu_0)^T\Sigma^{-1}(\mu_1-\mu_0)}}
$$
then $\mathbf w^T \mathbf x_0 = -(\gamma_1 - \gamma_0)$, and hence
$$
p(y = 1 | \mathbf x, \mathbf \theta) = sigmoid(\mathbf w^T(\mathbf x - \mathbf x_0))
$$
**So the final decision rule is as follows: shift $\mathbf x$ by $\mathbf x_0$, project onto the line $\mathbf w$, and see if the result is positive or negative**

![](/assets/LDA.png)

#### MLE for discrimiant analysis

To **fit a discriminant analysis model**, the simplest way is to **use maximum likelihood**.
$$
log~p(D|\theta) = [\sum_{i=1}^N\sum_{c=1}^CI(y_i = c)log\pi_c] + \sum_{c=1}^C[\sum_{i:y_i=c}logN(\mathbf x | \mathbf \mu_c, \Sigma_c)]
$$
For the class-conditional densities, we just partition the data based on its class label, and compute the MLE for each Gaussian
$$
\hat \mu_c = \frac 1 {N_c} \sum_{i:y_i=c}\mathbf x_i, \qquad \hat \Sigma_c = \frac 1 {N_c} \sum_{i:y_i=c} (\mathbf x_i - \hat {\mathbf \mu_c})(\mathbf x_i - \hat {\mathbf \mu_c})^T
$$

### Generative vs Discriminative classifiers 

When fitting a discriminative model, we usually maximize the conditonal log likelihood $\sum_{i=1}^Nlog~p(y_i|\mathbf x_i, \theta)$, whereas when fitting a generative model, we usually maximize the joint log likelihood $\sum_{i=1}^Nlog~p(y_i, \mathbf x_i | \theta)$

#### Pros and cons

* Easy to fit?: A naive Bayes and an LDA model can by fit by simple counting and averaging. By constrast, logistic regression requires solving a convex optimization problem
* Fit classes seperately: In a generative classifier, we estimate the parameters of each class conditional independently, so we do not have to retrain the model when we add more classes. In discriminative models, all the parameters interact, so the whole model must be retrained if we add a new class
* Can handle unlabeled training data: This is fairly easy to do using generative models, but is much harder to do with discriminative models
* Handle feature preprocessing: A big advantage of discriminative methods is that they allow us to preprocess the input in arbitrary ways, e.g., we can replace $\mathbf x$ with $\phi(\mathbf x)$. It is ofen hard to define a generative model on such pre-processed data, since the new features are correlated in complex ways
* Well-calibrated probabilities: Some generative models, such as naive Bayes, make strong independence assumptions which are often not valid. This can result in very extreme posterior class probabilities (very near 0 or 1). Discriminative models, such as logistic regression, are usually better calibrated in terms of their probability estimates