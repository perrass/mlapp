## Logistic Regression

### Intro

One way to build a probabilistic classifier is to **create a joint model of the form $p(y, \mathbf x)$ and then to condition on $\mathbf x$**, then **deriving $p(y|\mathbf x)$**. This is called the **generative approach**. Alternative approach is to **fit a model of the form $p(y|\mathbf x)$ directly**. This is called the **discriminative approach**

### Model specification

$$
p(y|\mathbf x, \mathbf w) = Ber(y|sigm(\mathbf w^T \mathbf x))
$$

### Model fitting

