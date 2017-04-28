## 	Linear Regression

### Intro

When augmented with kernels or other forms of basis function expansion, regression can model also non-linear relationship. When the Gaussian output is replaced with a Bernouli or multinouli distritbution

### Model specification

Linear regression
$$
p(y|\mathbf x, \theta) = N(y|\mathbf w^T \mathbf x, \sigma^2)
$$
Linear regression can be made to model non-linear relationships by replacing x with some non-linear function of the inputs.
$$
p(y|\mathbf x, \theta) = N(y|\mathbf w^T \phi(\mathbf x), \sigma^2)
$$
This is known as **basis function expansion**. A simple example are polynomial basis functions, where the model has the form 
$$
\phi(x) = [1, x, x^2, ..., x^d]
$$

### Maximum likelihood estimation (least squares)

A common way to esitmate the parameters of a statistical model is to compute the MLE, which is defined as 
$$
\hat \theta = argmax_\theta ~ log ~p(D|\theta)
$$
It is common to assume the training examples are independent and identically distributed (iid). This means we can write the **log-likelihood** as follows
$$
l(\theta) = log~p(D|\theta)=\sum_{i=1}^N~log~p(y_i|\mathbf x_i, \theta)
$$
Then we insert the Caussian into the above, and the log likelihood is
$$
\begin{align} l(\theta) & = \sum_{i=1}^N~log~[({1\over{2\pi\sigma^2}})^{1\over2}exp(-{1\over {2\sigma^2}}(y_i-\mathbf w^T \mathbf x_i))] \\
& = {-1\over{2\sigma^2}}RSS(\mathbf w) - {N\over2}log(2\pi\sigma^2)
\end{align}
$$
**RSS** stands for **residual sum of squres**, and is also called **sum of squared errors** or **SSE**, and SSS/N is called the **mean squared error** or **MSE**
$$
\begin{align} RSS(\mathbf w) &  = \sum^N_{i=1}(y_i-\mathbf w^T\mathbf x_i)^2 \\
& = ||\epsilon||^2
\end{align}
$$
where $\epsilon$ is residual errors $\epsilon_i = (y_i - \mathbf w^Tx_i)$

#### Derivation of MLE

$$
\begin{align} NLL(\theta) 
& = -l(\theta) \\
& = -\sum^N_{i=1}~log~p(y_i|\mathbf x_i, \mathbf \theta) \\
& = {1\over2}(\mathbf y - \mathbf X \mathbf w)^T(\mathbf y - \mathbf X \mathbf w) \\
& = {1\over2}\mathbf w^T(\mathbf X^T\mathbf X)\mathbf w - \mathbf w^T(\mathbf X^T\mathbf y)
\end{align}
$$

The gradient is 
$$
g(\mathbf w) = [\mathbf X^T\mathbf X\mathbf w - \mathbf X^T\mathbf y] = \sum^N_{i=1}\mathbf x_i(\mathbf w^T\mathbf x_i - y_i)
$$
Equating to zero we get the following equation. This is known as the **normal equation**. The corresponding solution $\mathbf {\hat w}$ to this linear system of equations is called the **ordinary least squares** or **OLS**
$$
\mathbf X^T \mathbf X \mathbf w = \mathbf X^T\mathbf y
$$

$$
\mathbf {\hat w_{OLS}} = (\mathbf X^T \mathbf X)^{-1}\mathbf X^T\mathbf y
$$

### Robust linear regression

One assumption of linear regression is that the residual error is normally distributed, but the data is usually with long tail. Hence, the model is not quiet well to fit outliers. There are two solution

1.  Using a more robust distribution like Laplace distribution, but the objective function $exp(-{1\over b}|y-\mathbf w^T x)$is non-linear and is hard to optimize. A **split variable** trick is to define $r_i = r_i^+ - r_i^-$, and then the problem is transformed to a **linear program** with D+2N unknowns and 3N constraints
2.  Using $-l(\theta)$ under a Laplace likelihood is to minimize the **Huber loss** function, which is everywhere differentiable 

$$
L_H(r, \delta) = \begin{cases} & r^2 / 2 ~~~\quad \qquad if |r| <= \delta \\
& \delta|r|-\delta^2/2 \quad if |r| > \delta
\end{cases}
$$

Consequently optimizing the Huber loss is much faster than using the Laplace likelihood, since we can use standard smooth optimization methods (such as quasi-Newton) instead of linear programming (using in solution one)

### Ridge Regression

The purpose of ridge regression is to overcome overfitting. And ridge regression is way to ameliorate this problem by **using MAP estimation with a Gaussian prior**.

The reason that the MLE can overfit is that it is picking the parameter values that are the best for modeling the training data; but if the data is noisy, **such parameters often result in complex functions**. To "smooth" the model, we use a zero-mean Gaussian prior:
$$
p(\mathbf w) = \prod_j~N(w_j|0, \tau^2)
$$
where ${1\over{\tau^2}}$ controls the strength of the prior. And the corresponding **MAP estimation problem** becomes
$$
argmax_{\mathbf w}\sum_{i=1}^N~logN(y_i|w_0+\mathbf w^T\mathbf x_i, \sigma^2)+\sum_{j=1}^Dlog~N(w_j|0, \tau^2)
$$
It is a simple exercise to show that this is equivalent to minimizing the following:
$$
J(\mathbf w) = {1\over N}\sum^N_{i=1}(y_i-(w_0+\mathbf w^T \mathbf x_i))^2 + \lambda||\mathbf w||^2_2
$$
where $\lambda = \sigma^2/\tau^2$ and $||\mathbf w||^2_2=\mathbf w^T\mathbf w$. **So both the variance of the assumed prior, $1 / \tau^2$, and the variance of the sample, $\sigma^2$,affect the complexity penalty**.

The solution is
$$
\hat {\mathbf w_{ridge}} = (\lambda \mathbf I_D + \mathbf X^T \mathbf X)^{-1}\mathbf X^T\mathbf y
$$
This technique is known as **ridge regression** or **penalized least squares**

**We will consider a variety of different priors in this book. Each of these corresponds to a different form of regularization**

### Bayesian Linear Regression (briefly)

Although **ridge regression** is a useful way to **compute a point estimate**, sometimes we want to **compute the full posterior over $\mathbf w$ and $\mathbf \sigma^2$**. In other words, ridge regression only returns **one line** or **a best line**, but bayesian linear regression returns **all sets of lines with probability**. ridge regression is one line in the sets, which owns the **maximum posterior**.

