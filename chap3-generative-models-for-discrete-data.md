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

**Toss coin**, a game involved **inferring a distribution** over a **discrete variable** drawn from a **finite hypothesis space**, given a **series of discrete observations**

#### Likelihood

If the data is iid, the likelihood has the form
$$
p(D|\theta) = \theta^{N_1}{(1-\theta)}^{N_0}
$$
where $\theta \in [0,1]$ is the probability of head, $N_1$ is the count of head and $N_0$ is the coutn of tail. These two counts are called the **sufficient statistics** of the data, due to $N = N_1 + N_0$

Suppose the data consists of the count of the number of heads $N_1$ observed in a fixed number $N = N_1 + N_0$ of trials. We have $N_1 \sim Bin(N, \theta)$, where $Bin$ represents the binomial distribution

#### Prior

For mathematical convenience, **the prior can have the same form as the likelihood**
$$
p(\theta) \sim \theta^{\gamma_1}{(1-\theta)}^{\gamma_2}
$$
Hence, the posterior is 
$$
p(\theta) \sim p(D|\theta)p(\theta) = \theta^{N_1}{(1-\theta)}^{N_0}\theta^{\gamma_1}{(1-\theta)}^{\gamma_2} = \theta^{N_1+\gamma_1}{(1-\theta)}^{N_0+\gamma_2}
$$
This **prior is a conjugate prior** for the corresponding likelihood. In the case of the Bernoulli, **the conjugate prior is the beta distribution**, the parameters of the prior are called **hyper-parameters**
$$
Beta(\theta|a, b) \sim \theta^{\alpha -1}{(1-\theta)}^{b-1}
$$

#### Posterior

$$
p(\theta | D) \sim Bin(N_1|\theta, N_0 + N_1)Beta(\theta |a, b) \sim Beta(\theta|N_1+a, N_0+b)
$$

The formular means the posterior is obtained by **adding the prior hyper-parameters to the empirical counts**. The **strength of the prior**, also known as the **effective sample size** of the prior, is $a + b$

If we add a single batch into the data with the notation, $D_a$, $D_b$ 

In batch mode, we have
$$
p(\theta | D_a, D_b) \sim Bin(N_1|\theta, N_0 + N_1)Beta(\theta |a, b) \sim Beta(\theta|N_1+a, N_0+b)
$$
where $N_1 = N^a_1 + N^b_1$ and $N_0 = N^a_0 + N^b_0$

**In sequential mode**, we have
$$
\begin{align} p(\theta|D_a, D_b) & \sim p(D_b|\theta)p(\theta|D_a) \\ & \sim Bin(N^b_1|\theta, N_1^b + N_0^b)Beta(\theta|N_1^a+a, N_0^a + b) \\ & \sim Beta(\theta | N_1^a + N_1^b + a, N_0^a+N_0^b + b)\end{align}
$$
This makes Bayesian inference particularly well-suited to **online learning**

### The Dirichlet-multinomial model



### Navie Bayes classifiers

