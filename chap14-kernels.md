# Chap14 Kernels

## Intro

Generally, the object that we wish to classify or cluster or process are represented as a **fixed-size feature vector**. However, what should we do if **it is not clear how to best represent them as fixed-sized feature vectors ?**

One approach is to define a **generative model** for the data, and use the **inferred latent representation** and/or the **parameters of models** as features, and then plug these features into standard methods. Deep Learning is a fantastic way to learn features

Another approach is to assume that we have some way of **measuring the similarity between objects, that doesn't require preprocessing them into feature vector format**. This approach uses **kernel function**

## Kernel functions

### RBF kernels

Gaussian Kernel is defined by
$$
k(\mathbf x, \mathbf x') = exp(-\frac 1 2 (\mathbf x - \mathbf x')^T\Sigma^{-1}(\mathbf x - \mathbf x'))
$$
If $\Sigma$ is diagnoal, this is known as **ARD kernel**, which means if $\sigma_j = \infty$, the corresponding dimension is ignored.
$$
k(\mathbf x, \mathbf x') = exp(-\frac 1 2 \sum^D_{j=1}\frac 1 {\sigma^2_j}(x_j - x'_j)^2)
$$
If $\Sigma$ is spherical, we get the isotropic kernel, where $\sigma^2$ is known as the **bandwidth**.
$$
k(\mathbf x, \mathbf x') = exp(-\frac {||\mathbf x - \mathbf x'||^2} {2\sigma^2})
$$
This is a example of a**radial basis function** or **RBF** kernel.

### Kernels for comparing documents

If we use a **bag of words representation**, where $x_{ij}$ is the number of times words $j$ occurs in document $i$, we can use **cosine similarity**
$$
k(\mathbf x_i, \mathbf x_i') = \frac {\mathbf x_i^T x_i'} {||\mathbf x_i||_2||\mathbf x_i'||_2}
$$
Drawbacks:

1. if they have many common words, like "the", "any", the two vectors are similar
2. if a discriminative word occurs many times in a document, the similarity is artificially boosted, even though word usage tends to be bursty, meaning that once a word is used in a document it is very likely to be used again. $\color{red}{?????}$

We use **TF-IDF**(term frequency inverse document frequency) to overcome drawbacks

1. **TF** is defined as a log-transform of the count: $tf(x_{ij}) = log(1 + x_{ij})$. This reduces the impact of words that occur many times within one document
2. **IDF** is defined as $idf(j) = log{\frac {N} {1 + \sum^N_{i=1}I(x_{ij}>0)}}$. N is the total number of documents, and the denominator counts how many documents contain term $j$
3. $\phi(\mathbf x) = TF \times IDF$

$$
k(\mathbf x, \mathbf x') = \frac {\phi(\mathbf x_i)^T \phi(\mathbf x_i')} {||\phi(\mathbf x_i)||_2||\phi(\mathbf x_i')||_2}
$$

### Mercer kernels

**Gram matrix**
$$
K = \begin{pmatrix} k(\mathbf x_1, \mathbf x_1) \cdots k(\mathbf x_1, \mathbf x_N)\\
\vdots \\
k(\mathbf x_N, \mathbf x_1) \cdots k(\mathbf x_N, \mathbf x_N)  
\end{pmatrix}
$$
Any set of inputs $\{ \mathbf x_i \}_{i=1}^N$. This kind of kernel is **Mercer kernel**

**Mercer theorem**

If the Gram matrix is positive definite, we can compute an **eigenvector decomposition** of it as follows:
$$
\mathbf K = \mathbf U^T\mathbf \Lambda\mathbf U
$$
where $\Lambda$ is a diagonal matrix of eigenvalues $\lambda_i > 0$. Now consider an element of $\mathbf K$
$$
k_{ij} = (\Lambda ^{\frac 1 2}U_{:,i})^T(\Lambda ^{\frac 1 2}U_{:,j})
$$
Then
$$
k_{ij} = \phi(\mathbf x_i)^T\phi(\mathbf x_j)
$$
Thus the entries in the kernel matrix can be computed by performing an inner product of some feature vectors that are implicitly defined by the eigenvectors $U$. In general, if the kernel is Mercer, the there exists a function $\phi$ mapping $\mathbf x \in X ~ to ~ \Bbb R^D$ such that
$$
k(\mathbf x, \mathbf x') = \phi(\mathbf x)^T \phi(\mathbf x')
$$
where $\phi$ depends on the eigen functions of $k$, so D is potentially infinite dimensional space

**Make new kernels**

New Mercel kernels can be derived from simple ones using a set of standard rules. E.g. $k_1$ and $k_2$ are both Mercer, so $k(\mathbf x, \mathbf x') = k_1(\mathbf x, \mathbf x') + k_2(\mathbf x, \mathbf x')$ is also Mercer.

### Linear kernels

$$
k(\mathbf x, \mathbf x') = \phi(\mathbf x)^T\phi(\mathbf x')
$$

**This is useful if the original data is already high dimensional, and if the original features are individually informative**. E.g., a bag of words or the expression level of many genes.

### Other kernels

There are other kernels, especially kernels from probabilistic generative models are also important.

E.g., The **string kernel** is equivalent to the **Fisher kernel** derived from an **L'th other Markov chain**. A kernel defined by the inner product of TF-IDF vectors is approximately equal to the **Fisher kernel** for a certain generative model of text based on the **compound Dirichlet multinomial model**  

## Using kernels inside GLMs

### Intro

We define a **kernel macine** to be a GLM where the input feature vector has the form
$$
\phi(\mathbf x) = [k(\mathbf x, \mathbf \mu_1), ..., k(\mathbf x, \mu_K)]
$$
where $\mu_K$ is a set of K **centroids**. This is **kernelized feature vector**. In this approach, the kernel need not to be a **Mercer kernel**

We can use the kernelized feature vector for logistic regression by defining $p(y|\mathbf x, \theta) = Ber(\mathbf w^T\phi(\mathbf x))$

Or, linear regression can be defined as $p(y|\mathbf x, \theta) = N(\mathbf x^T\phi(\mathbf x), \sigma^2)$

**How to choose the centroids $u_k$?**

A simple approach is to make each example $\mathbf x_i$ be a **prototype**, so we get
$$
\phi(\mathbf x) = [k(\mathbf x, \mathbf \mu_1), ..., k(\mathbf x, \mu_N)]
$$
Then, we can use any of the **sparsity-promoting priors for $\mathbf x$ to select a subset of the training examples**

**L1VM, L2VM** use l1, l2 regularization to subsampling

**RVM**(relevance vector machine) uses **ARD/SBL**

**SVM** modifies the likelihood term, rather than using a sparisity-promoting prior

### Sparsities of L1VM, L2VM, RVM, SVM

![](/assets/LogReg_Kernel.png)

The degree of sparsity is **RVM > L1VM > SVM > L2VM (No sparsity)**. For simplicity, we set $c = 1/\lambda$, where $\lambda$ is the regulation parameters of L1 and L2.

![](/assets/Reg_Kernel_Weights.png)

![](/assets/Reg_Kernel.png)

The degree of sparsity is **RVM > SVM > L1VM > L2VM (No sparsity)**

**How to derive a sparse kernel machine, or how to get sparsity?**

1. Using a GLM with kernel basis functions, plus a sparsity-promoting prior such as $l_1$ or ARD
2. Changing the negative log likelihood to some other loss function, such as SVM with Hinge loss or epsilon insensitive loss

## The kernel trick

Rather than defining our feature vector in terms of kernels, we can instead work with the original feature vectors $\mathbf x$, but modify the alogrithm so that it replaces all inner products of the form $<\mathbf x, \mathbf x'>$ with a call to the kernel function. This is called **kernel trick**. In this case, the **kernel should be Mercer kernel**.

### Kernelized KNN

Set $||\mathbf x_i - \mathbf x_i'||^2_2 = k(\mathbf x_i - \mathbf x_i) + k(\mathbf x_i' - \mathbf x_i') - 2k(\mathbf x_i - \mathbf x_i')$ 

This allows us use knn classifier to structured data objects.

### Kernelized ridge regression

**Primal problem** is to optimize $\mathbf w = (\mathbf X^T\mathbf X + \lambda \mathbf I_D)^{-1}\mathbf X^T\mathbf y$

**Dual problem**

Using matrix inversion lemma to rewrite the ridge estimate 
$$
\mathbf w = \mathbf X^T(\mathbf X \mathbf X^T+\lambda\mathbf I_N)^{-1}\mathbf y
$$
Then set a **dual variable**:
$$
\alpha = (\mathbf K + \lambda \mathbf I)^{-1}\mathbf y
$$
Hence, $\mathbf w = \mathbf X^T\mathbf \alpha = \sum^N_{i=1}\alpha_i\mathbf x_i$

At test time, the prediction is the predictive mean
$$
\hat f(\mathbf x) = \mathbf w^T\mathbf x = \sum^N_{i=1}\alpha_i\mathbf x_i^T\mathbf x = \sum^N_{i=1}\alpha_ik(\mathbf x, \mathbf x_i)
$$
**Time cost**

The time cost of primal variables is $O(D^3)$, and the dual variables $\alpha$ is $O(N^3)$. Hence, **kernel method can be useful in high dimensional settings**

However, prediction using the dual variables takes $O(ND)$ time, but using the primal variables only $O(D)$ time. We can speedup prediction by making $\alpha$ sparse.

## SVMs

The objective function, $l2$ regularized empirical risk function
$$
J(\mathbf w, \lambda) = \sum^N_{i=1}L(y_i, \hat y_i) + \lambda||\mathbf w||^2
$$

### Regression

Using epsilon insensitive loss, which is a variant of the Huber loss function
$$
L_{\epsilon}(y, \hat y) = \begin{cases} 0 \qquad if |y-\hat y| < \epsilon \\
|y - \hat y| - \epsilon \quad otherwise
\end{cases}
$$
This means that any point lying inside an $\epsilon-tube$ around the prediction is not penalized, hence
$$
J(\mathbf w, \lambda) = C\sum^N_{i=1}L_{\epsilon}(y_i, \hat y_i) + \lambda||\mathbf w||^2
$$
where $C = \frac 1 \lambda$. This objective is **convex and unconstrained, but not differentiable**

We formualte the problem as a **constrained optimization problem**. In particular, we introduce **slack variables (松弛变量)** to represent **the degree to which each point lies outside the tube**.
$$
y_i \le f(\mathbf x_i) + \epsilon + \xi^+_i \\
y_i \ge f(\mathbf x_i) - \epsilon - \xi^+_i
$$
Then, the objective is
$$
J = C\sum^N_{i=1}(\xi^+_i + \xi^-_i) + \frac 1 2 ||\mathbf w||^2
$$
This is a standard quadratic program in $2N + D + 1$ variables with linear constraints $\xi^+_i \ge 0$ and $\xi^-_i \le 0$. **PS: if $ |y-\hat y| < \epsilon$, $\xi^+_i = 0$ and $\xi^-_i = 0$**

Furthermore, the optimal solution has the form $\hat {\mathbf w} = \sum_i \alpha_i \mathbf x_i$, where $\alpha_i \ge 0$. And the $\alpha$ vector is sparse, because we don't care about errors which are smaller than $\epsilon$. The $\mathbf x_i$ for which $\alpha_i > 0$ are called the **support vectors**

### Classification

If we assume the labels are $y \in \{1, -1\}$, we can set the log loss is $L(y, \eta) = log(1+e^{-y\eta})$. This has a **convex upper bound on the 0-1 risk of a binary classifier**. In this case, we use hinge loss
$$
L_{hinge}(y, \eta) = max(0, 1 - y\eta) = (1 - y\eta)_{+}
$$
This is also non-differentiable, and we introduce slack variables $\xi_i$
$$
min_{\mathbf w, w_0, \xi} \frac 1 2 ||\mathbf w||^2 +C\sum^N_{i=1}\xi_i \qquad s.t. \xi_i \ge0, y_i(\mathbf x_i^T\mathbf w + w_0) \ge 1 -\xi_i, i=1,...,N
$$
If we use **Lagrange multipliers** as constraints, we can **eliminate** $\mathbf w, w_0, \xi$. Standard solvers take $O(N^3)$ time, but there exists specific alogrithms, **sequential minimal optimization (SMO)**, to decrease the cost of time to $O(D^2)$. Especially, we use linear SVM to decrease the cost time further to **O(N)**, if N is large.

The dual optimal solution has the form $\hat {\mathbf w} = \sum_i{\alpha_i \mathbf x_i} $. The $\alpha_i$ is sparse due to the hinge loss

At test time, prediction is done using
$$
\hat y(\mathbf x) = sgn(\hat w_0 + \sum^N_{i=1}\alpha_ik(\mathbf x_i, \mathbf x))
$$
It takes $O(sD)$ time to compute, where $s$ is the number of support vectors. In addtion, this means **SVM classifier produces a hard-labeling, but usually we want a probabilistic output to measure the confidence in our prediction**. This is the drawback of SVM.

**The large margin principle**

The main idea is in 《机器学习技法》by NTU. We discuss briefly here. The optimization is 
$$
min_{\mathbf w, w_0}\frac 1 2 ||\mathbf w||^2 \qquad s.t. y_i(\mathbf w^T\mathbf x_i + w_0) \ge 1, i = 1,...,N
$$
But we cannot ensure the data is linear seperable, hence cannot ensure $y_if_i \ge 1$ for all i. Hence, we introduce slack variable $\xi_i \ge 0$ such that

* $\xi_i=0$ if the point is on or inside the correct margin
* $0 < \xi_i \le 1$ if the point lies inside the margin, but on the correct side of the decision boundary
* $\xi_i >1$ if the point lies on the wrong side of the decision boundary

Then, we get
$$
min_{\mathbf w, w_0, \xi} \frac 1 2 ||\mathbf w||^2 +C\sum^N_{i=1}\xi_i \qquad s.t. \xi_i \ge0, y_i(\mathbf x_i^T\mathbf w + w_0) \ge 1 -\xi_i, i=1,...,N
$$

### Summary

In summary, SVM classifiers involve three key ingredients: **the kernel trick, sparsity, and the large margin principle**.

* The kernel trick is to **prevent underfitting** by **ensuring that the feature vector is sufficiently rich that a linear classifier can seperate the data**
* The sparsity and large margin principles are necessary to **prevent overfitting** by **ensuring not all the data used in basic functions**. Thest two ideas are closely related to each other and **both arise from the use of the hinge loss function** (for classification). 

However, there are other methods of achieving sparsity (such as $l_1$), and also other methods of maximizing the margin (such as **boosting**)

**PS, hinge loss introduces the large margin principle inherently. $1-y\eta$ is the margin, and we set $max(0,~margin)$**

$\color{red}{Questions}$

1. Does SVMs for regression use large margin principle, if this principle is introduced by hinge loss for classification
2. Why does boosting maximize the margin 

## Comparison of discriminative kernel methods

**The discussion will be written after finishing the chap13 and chap15**

If speed matters, use an RVM, but if well-calibrated probabilistic output matters, use a GP. The only circumstances under which using an SVM seems sensible is the **structured output case, where likelihood-based methods can be slow**

## Kernels for building generative models

