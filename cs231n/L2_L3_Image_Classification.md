# Image Classification

## Chanllenges

* Viewpoint variation: A single instance of an object can be oriented in many ways with respect to the camera (视角变化)
* Scala variation: Visual classes often exhibit variation in their size (尺寸放缩)
* Deformation: Many objects of interest are not rigid bodies and can be deformed in extreme ways (不正常的同类图片)
* Occlusion: The objects of interest can be occluded. Sometimes only a small portion of an object could be visible (被遮挡)
* Illumination conditions: The effects of illumination are drastic on the pixel level (光影)
* Background clutter: The objects of interest may blend into their environment, making them hard to identify (与背景接近)
* Intra-class variation: The classes of interest can often be relatively broad, such as chair (同类间差异过大，比如椅子)

## Nearest Neighbor

Calculate the distance (can be L1 or L2 norm), and then find the k nearest neighbor and using the majority of class of the neighbors as prediction

**Warning:** In image, the norm is **pixel-wise**

![](assets/knn.png)

### Pros & Cons

The pro is that the classifier takes no time to train, since all that is required is to store and possibly index the training data. However, the cons are that we pay that computational cost at test time, since classifying a test example requires a comparison to every single training example, and the distances metrics of high dimensional data is problematic. E.g., the distance of the following figures are the same. That is, the same distance or the same sum of a grouped data, may wrongly classify the images.

![](assets/misclassify_by_knn.png)

### Improvement

**Approximate Nearest Neighbor** algorithms allow one to trade off the correctness of the nearest neighbor retrieval with its space/time complexity during retrieval, and usually rely on a pre-processing/index stage that involves building a **kdtree**, or running the **k-means** algorithms. ([FLANN Lib](http://www.cs.ubc.ca/research/flann/))

## Parameterized Mapping

For example, in CIFAR-10 we have a trainging set of $N = 50000$ images, each with $D = 32 \times 32 \times3 = 3072$ pixels, and $K = 10$. Now we define the score function **$f: \Bbb R^D \to \Bbb R^K$ that maps the raw image pixels to class images**.

## Linear Classfication

$$
f(x_i, W, b) = Wx_i + b
$$

where $W$ with the size $K \times D$, and $b$ with the size $K \times 1$. In CIFAR-10, $x_i$ contains all pixels in the i-th image flattened into a single $3072\times 1$ column, $W$ is $10 \times 3072$, and $b$ is $10 \times 1$

**Notes:** The single matrix multiplication $Wx_i$ is effectively **evaluating 10 seperate classifiers in parallel**, 也就是说在梯度下降的时候矩阵操作可以并行，提升运算速度，矩阵乘法天然并行

## Multiclass Support Vector Machine

The SVM loss is set up so that the SVM "wants" the correct class for each image to a have a score higher than the incorrect classes by some fixed margin $\Delta$.
$$
L_i = \sum_{j \neq y_i} max(0, f(x_i, W)_j - f(x_i, W)_{y_i} + \Delta)
$$
Example:

The scores $s = [13, -7, 11]$, and that the first class is the true class, and $\Delta = 10$. Hence, 
$$
L_i = max(0, -7-13+10) + max(0, 11-13+10) = 8
$$
The function, $max(0, -)$ is called **hinge loss**. There also exists **the squared hinge loss** or L2-SVM, which uses the form $max(0,-)^2$ that **penalizes violated margins more strongly**

## Regularization

Rather than decreasing the magnitude of weights to decrease the variance of the model, there is another motivation of regularization.

**The issue is that this set of $\mathbf W$ is not necessarily unique: there might be many similar $\mathbf W$ that correctly classify the examples**. That is, if the loss is 0, $\mathbf W$ and $2\mathbf W$ are both 0. If we use L2 Regularization
$$
L = \frac 1 N \sum_i\sum_{j \neg y_i}[max(0, f(x_i; W)_j - f(x_i;W)_{y_i} + \Delta)] + \lambda \sum_k\sum_lW_{k,l}^2
$$
The most appealing property is that penalizing large weights tends to improve generalization, because it means that no **input dimension can have a very large influence on the scores all by itself**

This means that **with the same data loss, we choose a set of parameters with less regularization loss**. This leads to we want the distribution of the parameter to be more close to **uniform distribution**. This is the idea of **maxent** 

---

### Aside: Practical Considerations

#### Setting Delta

Practically, we set $\Delta = 1.0$ which is safe for all cases. These two parameters $\Delta$ and $\lambda$ are all regularizers, but with the same functionality (rather than L1 and L2, considering the tech called **elastic net**). The effect of change of $\Delta$, can be get from changing $\lambda$, because the weights can **shrink or stretch the differences arbitrarily**. Hence, **the only real tradeoff is how large we allow the weights to grow by tuning $\lambda$ **

#### Optimization in primal

In most machine learning materials, SVM is trained by the tricks of **kernels, duals, or the SMO algorithm**. However, in this class, we will always work with the optimization objectives in their **unconstrained primal form**. Many of these objectives are technically not differentiable, but in practice this is not a problem and it is common to use a **subgradient**

#### Other Multiclass SVM formulations

Multiclass SVM in one way of formulating the SVM over multiple classes. This method is belong to **All-vs-All** strategy. There are other two strageties: **One-vs-All** strategy and **Structured SVM**

For **One-vs-All**, MLAPP quotes

> We train $C​$ binary classifiers, $f_c(\mathbf x)​$, where the data from class $c​$ is treated as positive. and the data from all the other classes is treated as negative. A particular point is assigned to the class for which the distance from the margin, in the positive direction (i.e., in the direction in which class "one" lies rather than class "rest". However, this can result in regions of input space which are **ambiguously labeled** (i.e., both have class $c1​$ and $c2​$)

Alternatively,

> We pick $\hat y(\mathbf x) = argmax_C f_c{(\mathbf x)}$. However, this technique may not work either, since there is no guarantee that the different $f_c$ functions have comparable magnitudes. In addition, each binary subproblem is likely to suffer from the **class imbalance** problem. That is, if we have 10 classes which have uniform distribution, the percent of true label is $10\%$, and that of flase label is $90\%$

---

## Softmax

If we change the hinge loss from SVM to **cross-entropy** loss, then the classifier is changed to softmax. The cross-entropy loss is 
$$
L_i = -log(\frac {e^{f_{y_i}}} {\sum_j e^{f_j}}) \quad or, equivalently\quad L_i = -f_{y_i}+log\sum_j e^f_J
$$

#### Information theory view

The cross-entropy between a true distribution $p$ and an estimated distribution $q$ is defined as 
$$
H(p, q) = \sum_xp(x)logq(x)
$$
The softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ($q = e^{f_{y_i}}/\sum_je^{f_j}$) and the true distribution $p$. Moreover, since the cross-entropy can be written in terms of **entropy and KL  divergence** as $H(p, q) = H(p) + D_{KL}(p||q)$. **The entropy of the delta function $p$ is zero, this is also equivalent to minimizing the KL divergence between the two distributions (a measure of distance)**. In other words, the cross-entropy objective wants to the predicted distribution to have all of its mass on the correct answer

**Probabilistic interpretation**
$$
P(y_i|x_i, W) = \frac {e^{fy_i}} {\sum_j e^{f_j}}
$$
is the normalized probability assigned to the correct label $y_i$ given the image $x_i$ and parameterized by $W$. Probabilistically, we add a Gaussian prior over the weight matrix $W$ in terms of $L2$ regularization, then we find the negative log likelihood function of the correct class, and then we optimize the function and get the Maximum a posteriori estimation, which has the optimized $W$

#### Practical issues: output probabilities of softmax

Considering to output probabilities: 
$$
\begin{align}
[1, -2, 0] \to [e^1, e^{-2}, e^0] \to [2.71, 0.14, 1] \to [0.7, 0.04, 0.26]  & \quad (1)\\
[0.5, -1, 0] \to [e^{0.5}, e^{-1}, e^0] \to [1.65, 0.37, 1] \to [0.55, 0.12, 0.33] & \quad (2)
\end{align}
$$
The penalty parameter $\lambda$ of (2) is the half of that of (1), and this leads to the different probabilities. This means that **the ordering of the scores is interpretable, but the absolute numbers technically are not**

#### Practical issues: hinge and cross-entropy loss

Considering three inputs:
$$
\begin{align}
[10, -2, 3] & \quad (1)\\
[10, -100, -100] & \quad (2)\\
[10, 9, 9] & \quad (3)
\end{align}
$$
In situation (1), with $\Delta = 1$, there is no big difference between hinge loss and cross-entropy. However, **the hinge loss cannot distinct the difference between (2) and (3), but the cross-entropy function can**

## Structured Learning 

### Definition

Input and output are both objects with structures (e.g., sequence, list, tree, bounding box,...)

* Speech recognition: Speech signal -> text (sequence -> sequence)
* Translation: Mandarin -> English (sequence -> sequence)
* Object Detection: Image -> **bounding box** 
* Summarization:  Long doc ->  summary (sequence -> sequence)
* Retrieval: Keyword -> search result (a list of pages)

Formally, in training, we find a function $F$
$$
F: X \times Y \to R
$$
where $F(X, Y)$: evaluate how compatible the objects $x$ and $y$ is, $R$ is a real value to measure the compatibility

In testing, given an object $x$, we get $y$ by
$$
\hat y = argmax_{y\in Y}F(x, y)
$$
Furthermore, in training, we estimate the probability $P(x, y)$
$$
P: X \times Y \to [0, 1]
$$
In testing, given an object $x$
$$
\begin{align}
\hat y & = argmax_{y\in Y}P(y|x) \\
& = argmax_{y\in Y}\frac {P(x,y)} {P(x)} \\ 
& = argmax_{y\in Y}P(x, y)
\end{align}
$$
 也就是，训练一个$(x,y)$的分布，预测时给出x时，y是概率最高的y	

### How to structured learning

#### Problem 1

What does $F(x, y)$ look like?

Assume the linear function, we have $F(x,y) = w_1\phi(x_1,y_1) + ... + w_n\phi(x_n, y_n) = \mathbf w\phi(\mathbf x, \mathbf y)$

#### Problem 2

How to solve the argmax function $y = argmax_{y\in Y}F(x, y)$

For object detection, we can use **Branch and Bound** algorithm or **Selective Search**

For Sequence Labeling, we can use **Viterbi** Algorithm

#### Problem 3

Training: Given training data, how to learn $F(x, y)$. Practically, $F(x, y) = \mathbf w\phi(\mathbf x, \mathbf y)$ and for all training examples $\forall r$, and all incorrect label for r-th example, $\forall y \in Y - \{\hat y^r\}$ we have
$$
\mathbf w\cdot \phi(x^r, \hat y^r) > \mathbf w\cdot \phi(x^r, y)
$$

#### Algorithm

* Input: training data set $\{(x^1, \hat y^1), ..., (x^r, \hat y ^r)\}$
* Output: weight vector $w$
* Algorithm: Initialize $w = 0$
  * Do
    * For each pair of training example $(x^r, \hat y^r)$
      * Find the label $\tilde y^r$ maximizing $w\cdot \phi(x^r, y)$
        * $\tilde y^r = argmax_{y\in Y} w\cdot \phi(x^r, y)$
      * If $\tilde y^r \ne \hat y^r$, update $w$
        * $w\to w + \phi(x^r, \hat y^r) - \phi(x^r, \tilde y^r)$
  * Until $w$ is not updated

### Structured SVM

