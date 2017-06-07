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

### Statistical Significance

### Pooled Standard Error

### Sensitivity

## Policy and Ethics for Experiments

## Choosing and Characterizing Metrics

## Designing an Experiment

## Analyzing Result

## Final Project

