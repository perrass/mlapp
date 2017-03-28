## Generative models for discrete data

Generative Bayesian Classifier
$$
p(y = c | \mathbf x, \mathbf \theta)  \sim p(\mathbf x | y = c, \theta)p(y = c | \theta)
$$
In this chap, we will discuss discrete data and **how to infer the unknown parameter $\theta$ of such model$$

### Bayesian concept

**Posterior predictive distribution**: $p(\hat x | D)$, which is the probability that $\hat x \in C$ given the data $D$ for any $\hat x \in [1,...,100]$

**Hypothesis space**: a assumption, or a concept. E.g. even number, power of two

**Likelihood**: $p(D|h)$, which is the probability of $D$, given a hypothesis. E.g $D = [16]$, $H$ is that data is the power of two, the population of $D$ is $[1-100]$. Hence $p(D|h) = p(x = 16 | [2, 4, 8, 16, 32, 64]) = 1 ~ / ~ 6$, or $p(D|h_{even}) = 1 \ / \ 50$

**Prior** is the mechanism by which background knowledge can be brought to bear on a problem

**Posterior** is the **normalized result of the times of the likelihood and the prior** 
$$
p(h|D) = {{p(D|h)p(h)}\over{\sum_{h' \in H} p(D, h')}}
$$
In general, when we have enough data, the posterior $p(h|D)$ becomes peaked on a single concept, **MAP estimate**
$$
{\hat h}^{MAP} = argmax_hp(D|h)p(h) = argmax_h[log\ p(D|h) + log\ p(h)]
$$
Since the likelihood term depends exponentially on N, and the prior stays constant, as we get more and more data, the MAP estimate converges towards the **maximum likelihood estimaite**. Hence, **if we have enough data, we see that the data overwhelms the prior**

### The beta-binomial model

**Toss coin**

#### Likelihood

### The Dirichlet-multinomial model



### Navie Bayes classifiers

