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

## The kernel trick

## SVMs

## Comparison of discriminative kernel methods

## Kernels for building generative models

