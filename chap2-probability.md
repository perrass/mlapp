## Probability

### Intro

* **frequentist**: probabilities represent long run frequencies of events
* **bayesian**: probability is used to quantify our uncertainty about something

### Brief review of probability theory

The expression $p(A)$ denotes the probability that the event A is true. Here $p()$ is called a **probability mass function** or **pmf**

#### Fundamental rules

1. union $p(A\cup B) = p(A) + p(B) - p(A \cap B)$
2. joint $p(A, B) = p(A\cap B) = p(A|B)p(B)$ (**product rule**)
3. marginal distribution $p(A) = \sum_b p(A,B) = \sum_bp(A|B= b)p(B=b)$ (**sum rule**)
4. **chain rule** $p(X_{1:D}) = p(X_1)p(X_2|X_1)...p(X_D|X_{1:D-1})$
5. **conditional probability** $p(A|B) = {p(A,B)\over p(B)}\ if\ p(B)\ >\ 0$

**Bayes rule**
$$
p(X = x | Y = y) = {p(X = x, Y = y)\over p(Y = y)} = {{p(X=x)p(Y = y|X = x)\over{\sum_{x'}p(X=x')p(Y = y|X=x')}}}
$$

* The numerator is the product rule
* The denominator is the sum rule

**Independence and conditional independence**

* $p(X, Y) = p(X)p(Y)$
* $p(X, Y|Z) = p(X|Z)p(Y|Z)$

Assuming that X has 6 possible status and Y has 5 possible status, the general joint distribution on two such variables would require **29** parameters. If X and Y are unconditional independence, only **(6-1) + (5-1) = 9** parameters to define p(x, y).

#### Continuous random variables

* **cumulative distribution function** of X is defined as $F(q) = p(X <= q)$ , and $p(a < X <= b) = F(b) - F(a)$
* **probability density function** is defined as $f(x) = {d\over {dx}}F(x)$

#### Quantiles

Since the cdf F is a monotonically increasing function, it has an inverse $(F^{-1})$. If F is the cdf of X, then $F^{-1}(\alpha)$ is the value of $x_{\alpha}$ such that $P(X <= x_{\alpha}) = \alpha$, this is called the $\alpha$ quantile of F 

### Discrete distributions 

* binomial 
* Bernoulli (a special case of Binomial distribution with n = 1)
* Poisson
* Empirical

### Continuous distributions

* Gaussian,  not robust to outliers
* Student t, robust to outliers
* **Laplace**, robust to outliers
* gamma
  * exponential
  * erlang
  * **Chi-squared**, $Ga(x|{v\over2}, {1\over2})$
  * $\Gamma(x) = \int_0^\infty u^{x-1}e^{-\mu}d\mu$
* beta, is also based on **gamma function** ($\Gamma(\alpha)$)
* pareto

### Joint probability distributions



### Transformations of random variables



### Monte Carlo approximations



### Information theory

