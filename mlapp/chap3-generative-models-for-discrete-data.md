## Generative models for discrete data

Generative Bayesian Classifier
$$
p(y = c | \mathbf x, \mathbf \theta)  \sim p(\mathbf x | y = c, \theta)p(y = c | \theta)
$$
In this chap, we will discuss discrete data and **how to infer the unknown parameter $\theta$ of such model**

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

**The probability of heads is $\theta$**

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

The MAP estimate is given by, (due to Beta distribution)
$$
\hat \theta_{MAP} = {{a+N_1-1}\over{a+b+N-2}}
$$
If we use a uniform prior, the MAP estimate reduces to MEL
$$
\hat \theta_{MLE} = {N_1\over N}
$$
The posterior mean is 
$$
\overline \theta = {{a + N_1} \over {a + b + N}}
$$
The posterier mean is **convec combination** of the **prior mean** and the **MLE**, which captures the notion that **the posterior is a compromise between what we previously believed and what the data is telling us**

**Proof**

Let $\alpha_0 = a + b$ be the **equivalent sample size** of the prior, which **control its strength**, and let the prior mean be $m_1 = a / \alpha_0$. Then the posterior mean is given by
$$
E(\theta|D) 
={{\alpha_0m_1 + N_1}\over{N + \alpha_0}} 
= {\alpha_0\over{N+\alpha_0}}m_1 + {N\over{N + \alpha_0}}{N_1\over N} 
= \lambda m_1 + (1-\lambda)\hat \theta_{MLE}
$$
where $\lambda = {\alpha_0\over{N+\alpha_0}}$ is the ratio of the prior to posterior equivalent sample size.The weaker the prior, the smaller is $\lambda$

**In sequential mode**, we have
$$
\begin{align} p(\theta|D_a, D_b) & \sim p(D_b|\theta)p(\theta|D_a) \\ & \sim Bin(N^b_1|\theta, N_1^b + N_0^b)Beta(\theta|N_1^a+a, N_0^a + b) \\ & \sim Beta(\theta | N_1^a + N_1^b + a, N_0^a+N_0^b + b)\end{align}
$$
This makes Bayesian inference particularly well-suited to **online learning**

#### Posterior predictive distribution

The probability of heads in a single future trial under a $Beta(a, b)$ posterior, we have
$$
p(\hat x = 1 |D) 
= \int_0^1p(x = 1 | \theta)p(\theta|D)d\theta
= \int_0^1\theta Beta(\theta|a, b)
= E(\theta|D)
= {a\over {a+b}}
$$
In this case, the posterior predictive distribution is equivalent to plugging in the posterior mean parameters

However, if we have a small sample, e.g. $N= 3$ tails in a row. The posterior of head is 0. This is called **sparse data problem**, or in intuition, the **black swan**

**Add-one smoothing** or **Laplace's rule of succession** can solve this.

### The Dirichlet-multinomial model



### Navie Bayes classifiers

