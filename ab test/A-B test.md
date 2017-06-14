# A/B Testing

## Overview of A/B Testing

### A/B Testing Cases

1. Online shopping company want to know: does my website provide all products consumers want to buy? $\color{red}{No}$
2. Add permium service $\color{red}{No}$
3. Implement a new ranking algorithm for a movie recommendation site $\color{red}{Yes}$
4. Change the backend, so the page load time and results users see would be changed $\color{red}{Yes}$
5. Website selling cars, will a change increase repeat customers or referals $\color{red}{No}$ -> The reason is that,the behavior of users in this website is in low-frequency, in this case A/B test is not suitable.
6. Update brand, including the main logo $\color{red}{No}$ -> The reason is that this affect is suprisingly emotional, so the data collect in a short time cannot support to judge whether this exchage is good or not
7. Test layout of initial page $\color{red}{Yes}$

### Audacity Example

**Customer funnel**: Homepage visits -> Exploring the site -> Creating account -> Finishing a course 

Metric

**Business Metrics:** total number of cources completed

**Statistical Metrics:**

* click-through rate: num of clicks / num of page views
* click-through probability: num of unique visitors who click / num of unique visitors to this page

### Confidence Intervals

Before talking about confidence intervals, we assume the target is **binomially distributed**, which means we test whether the user click or not

**Question**: there are 1000 unique users, and 100 users click the button. If we change the color of button, how many users will click the button in general case? What is the interval?

**Answer**:

The point probability is 100 / 1000 = 0.1

If we want 95% confidence, the **z-score** is [-1.96, 1.96].

The standard deviation of binomial distribution in this case is $\sqrt{\hat p (1 - \hat p) / N} = \sqrt{0.1 \times 0.9 / 1000} = 0.009486$

$m = z-score * sd = 0.019$

Hence, the interval probability is [0.081, 0.119], and if there more than 119 users or less than 81 users who click the button, we would be surprised.

### Statistical Significance (Hypothesis Test)

1. Set the **null hypothesis** and **alternative hypothesis**
2. Choose the appropriate **statistical value and corresponding statistical test**

**E.g**

P(result due to chance), we set two groups, P_control and P_experiment.

The null hypothesis is P_experiment - P_control = 0, which means there is no change

The alternative is P_experiment - P_control > 0, which means there is a change

We set a statistical criterion, such as 0.05, and if p < 0.05, we reject null hypothesis, and the experiment is statistically significant

**E.g. Pooled Standard Error**

We have X_cont, X_exp, which are samples, and N_cont, N_exp, which are populations
$$
\hat p_{pool} = (x_{cont} + x_{exp}) / (N_{cont} + N_{exp})
$$
The standard deviation of this binomial distribution is 
$$
SE_{pool} = \sqrt{\hat p_{pool} \times (1 - \hat p_{pool}) \times (\frac 1{N_{cont}} + \frac {1} {N_{exp}}) }
$$
The difference is $\hat d = \hat p_{exp} - \hat p_{cont}$, and the null hypothesis is $\hat d = 0, d \sim N(0, SE_{pool})$

If we set $\alpha = 0.05$ and choose Z-test, the criterion is $\hat d > 1.96 \times SE_{pool}$ or $\hat d < -1.96 \times SE_{pool}$, we reject the null

### Sensitivity

|                | Test result: True                      | Test result: False                     |
| -------------- | -------------------------------------- | -------------------------------------- |
| Reality: TRUE  | True Positive (1 - $\beta$)            | False Negative $\beta$ (Type II error) |
| Reality: FALSE | False Positive $\alpha$ (Type I error) | True Negative (1 - $\alpha$)           |

The level of sensitivity(power) determines how many samples we need to calculate the A/B test. In [Sample size of A/B test](http://www.evanmiller.org/ab-testing/sample-size.html), we can calculate the number.

![](/assets/Sample_size.png)

### Overview

**E.g.**

N_cont = 10072, N_exp = 9886, X_cont = 974, X_exp = 1243, d_min = 0.02, confidence level = 0.95. Would you launch the new change?

**Answer**
$$
\begin{align} \hat p_{pool} & = (974 + 1242) / (10072+9886) & = 0.111 \\
	SE_{pool} & = \sqrt{0.111(1-0.111)(1/10072 + 1/9886)} & = 0.00445 \\
	\hat d & = 1242 / 9886 - 974 / 10072 & = 0.0289\\
	m & = \alpha * SE_{pool} & = 0.0087
\end{align}
$$
Hence, $\hat d_{min} = 0.0202, \hat d_{max} = 0.0376$ and we should launch the change.

![](/assets/Confidence_Interval.png)

## Choosing and Characterizing Metrics

### Redefine the customer funnel

1. Homepage visits
2. Course list visits
3. Visits to course pages
4. Account created
5. Enrollement
6. Coaching use
7. Complemetation
8. Jobs

Each stage is a metric: the number of users who reach that point, rates or probabilities 

## Designing an Experiment

## Analyzing Result

## Final Project

