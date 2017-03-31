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

#### Generative classifiers

$$
p(y=c|\mathbf x, \theta) = {{p(y=c|\theta)p(x|y=c,\theta)}\over{\sum_{c'}p(y=c'|\theta)p(\mathbf x|y=c', \theta)}}
$$

It specifies how to generate the data using **class-conditional density** $p(\mathbf x | y = c)$, and the class prior $p(y=c)$

Another approach is to **directly fit the class posterior**, $p(y=c|\mathbf x)$

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

A **joint probability distribution** has the form $p(x_1,...x_D)$ for a set of $D > 1$ variables, and models the (stochastic) relationships between the variables. However, the **number of parameters** needed to define such a model is $O(K^D)$, where K is the number of states for each variable. By making **conditional independence assumptions**, the params would be fewer.

**Covariance** between two rv's X and Y measures the degree to which **X and Y are (linearly) related**, and the range is 0 to infinity.
$$
cov[X,Y] = E[(X-E[X])(Y-E(Y)] = E[XY]-E[X]E[Y]
$$
**Correlation coefficient** normalized the measure, a degree of linearity
$$
corr[X,Y] = {cov[X,Y]\over{\sqrt {var[X]var[Y]}}}
$$
If X and Y are independent, meaning $p(X,Y) = p(X)p(Y)$, then $cov[X,Y]=0$, and $corr[X,Y]=0$, so they are uncorrelated. However, **uncorrelated does not imply independent**. 

**A more general measure of dependence bewteen random variables is *mutual information***

* Multivariate Gaussian
  * A full covariance matrix has $D(D+1)/2$ params
  * A diagonal covariance has $D$ params
  * A spherical or isotropic covariance, $\Sigma = \sigma^2I_D$, has one free param
* Multivariate Student t 
* Dirichlet distribution is the multivariate generalization of the beta distribution 

### Transformations of random variables

If x ~ p() is some random variable, and $y = f(x)$, what is the distribution of y ?

### Monte Carlo approximations

1. Generate $S$ samples from the distributions (for high dimensinoal distributions, MCMC, is often used)
2. Given the samples, we can approximate the distribution of $f(X)$ by using the empirical distribution of ${f(x_s)}^S_{s=1}$ 

### Information theory

Informatino theory is concerned with representing data in a compact fashion (a task known as data compression or source coding), as well as with transmitting and storing it in a way that is robust to errors.

#### Entropy

The **entropy** of a random variable $X$ with distribution $p$ is a measure of its **uncertainty**
$$
H(x) = -\sum^K_{k=1}p(X=k)log_2p(X=k)
$$
The discrete distribution with maximum entropy is the uniform distribution. Hence, for a K-ary random variable, the entropy is maximized if $p(x = k) = {1\over K}$ 

For binary random variables, $X \in {0, 1}$, $p(X = 1) = \theta$ and $p(X=0)=1-\theta$, so entropy becomes the following, and the maximum is 1 when $\theta = 0.5$
$$
\begin{align} H(x) & =  -[p(X=1)log_2p(X=1) + p(X=0)log_2p(X=0)] \\ & = -[\theta log_2\theta + (1-\theta)log_2(1-\theta)] \end{align}
$$

#### KL divergence

**Kullback-Leibler divergence** or **relative entropy** measures the **dissimilarity** of two **probability distritbutions**, $p$ and $q
$$
\begin{align} KL(p||q) & = \sum^K_{k=1}p_k log{p_k\over q_k}\\ & =  \sum_kp_klog\ p_k - \sum_kp_klog\ q_k \\ & = -H(p) + H(p, q)\end{align}
$$
where $H(p,q)$ is called the **cross entropy**
$$
H(p,q) = -\sum_k p_k log\ q_k
$$
Cross entropy is the average number of bits needed to encode data coming from a source with distribution $p$ when we use model $q$ to define our codebook. While, KL divergence is the average number of **extra** bits needed to encode the data, due to the fact that we used distribution $q$ to encode the data instead of the true distribtion $p$ 

Hence, this leads to $KL(p||q) >=0$, and that the KL is only equal to zero iff $q = p$ (also leads to **Laplace's principle of insufficient reason**)

#### Mutual information

Consider two random variables X, and Y. Mutual information determine **how similar the joint distribution $p(X,Y)$ is to the factored distribution $p(X)p(Y)$**. This interprate how much knowing one variable tells us about the other.
$$
I(X, Y) = KL(p(X, Y)||p(X)p(Y)) = \sum_x\sum_yp(x,y)log{p(x,y)\over {p(x)p(y)}}
$$
We have $I(X,Y) >= 0$ with equality iff $p(X,Y) = p(X)p(Y)$. That is, the MI is zero iff the **variables are independent**

Also,
$$
I(X,Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
$$
where $H(Y|X)$ is the **conditional entropy**, defined as $H(Y|X) = \sum_xp(x)H(Y|X = x)$. Hence, MI between X and Y as the **reduction in uncertainty about X after observing Y**

### Notes

* The KL divergence **is not a distance**, since it is **asymmetric**. One sysmmetric version of the KL divergence is the **Jensen-Shannon divergence**. $JS(p_1, p_2) = 0.5KL(p_1||q) + 0.5KL(p_2||q)$ , where $q = 0.5p_1 + 0.5p_2$