# Neural Networks I: Setting up the Architecture

## Activation functions

### Sigmoid

Drawbacks:

1. **Sigmoids saturate and kill gradients**: when the neuron's activation saturates at either tail 0 or 1, the gradient at these regions is almost zero. In addtion, the max gradient of sigmoid is 0.25. Hence, the gradients would decline or even be killed
2. **Sigmoid outputs are not zero-centered**: this has implications on the dynamics during gradient descent, because if the data coming into a neuron is always positive, then **the gradient on the weights $w$ will during backpropagation become either all be positive, or all negative**. This could introduce **undesirable zig-zagging dynamics in the gradient updates** for the weights. However, notice that once these gradients are added up across a batch of data the final update for the weights can have variable signs, somewhat mitigating this issue.

### Tanh

$$
tanh(x) = 2\sigma(2x) -1
$$

### ReLU

$$
f(x) = max(0, x)
$$

Pros

1. Greatly accelerate the **convergence of stochastic gradient descent** compared to the sigmoid/tanh functions. It is argued that this is due to its **linear, non-saturating** form
2. non-expensive operations

Cons

1. ReLU units can be **fragile during training and can die**. For example, a large gradient flowing through a ReLU neuron could cause the weights to update in such a way that the neuron will never activate on any datapoint again. If this happens, then the gradient flowing through the unit will forever be zero from that point on

### Leaky ReLU

$$
f(x) = 1(x<0)(\alpha x) + 1(x \ge 0)(x )
$$

where $\alpha$ is a small constant. 

This activation function solves the "dying ReLU" problem

## Represetational Power

**The one hidden layer neural network can approximate any continuous function**. In practice, the case that 3-layer neural networks will outperform 2-layer nets, but going even deeper rarely helps much more.

## Data Preprocessing

1. **Mean substraction**
2. **Normalization**:
   1. Divide each dimension by its standard deviation, once it has been zero-centered
   2. Normalize each dimension so that the min and max along the dimension is -1 and 1 respectively
3. **PCA and Whitening**

We mention PCA/Whitening in these notes for completeness, but these transformations are not used with Convolutional Networks. However, **it is very important to zero-center the data, and it is common to see normalization of every pixel as well**.

**Common pitfall**: The mean must be computed only over the training data and then subtracted equally from all splits (train/val/test)

## Weight Initialization

```python
w = np.random.randn(n) * sqrt(2.0 / n)
bias = np.zeros(n)
```

This is the current recommendation for use in practice in the specific case of neural networks with ReLU neurons

## Regularization

L1/L2 and Elastic Net and Dropout

### Max norm constraints

In practice, this corresponds to performing the parameter updates as normal, and then enforcing the constraints by clamping the weight vector $\hat w$ of every neuron to statisfy $||\hat w||_2 <c$. Typical values of $c$ are on orders of 3 or 4.

### Theme of noise in forward pass

During forward pass, set a group of weights to zero. E.g. **stochastic pooling, fractional pooling, and data augmentation**

In practice, we use a single, global L2 regularization by cross-validation. And dropout

## Loss

### Large number of classes

When the set of labels is very large, it may be helpful to use **Hierachical Softmax**. The hierachical softmax decomposes labels into a tree.  Each label is then represented as a path along the tree, and a Softmax classifier is trained at every node of the tree to disambiguate between the left and right branch. The structure of the tree strongly impacts the performance and is generally problem-dependent.

### Regression

The most popular form of regression is **mean loss square**, but it is much harder to optimize than a more stable loss such as softmax.

1. The magnitude of softmax does not influence the result, but those value of regression would be influent
2. The L2 loss is less robust because outliers can introduce huge gradients
3. Applying dropout is not a good idea

**When we faced with a regression task, first consider if it is absolutely necessary. Instead, have a strong perference to discretizing your outputs to bins and perform classification over them whenever possible**

