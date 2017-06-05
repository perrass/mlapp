# Lesson3 Partition-Based Clustering Methods

Basic Idea: Partitioning a dataset D of n objects into a set of K clusters so that an objective function is optimized.

A typical object function is: **Sum of Squared Error**
$$
SSE(C) = \sum^K_{k=1}\sum_{x_{i\in C_k}}||x_i-c_k||^2
$$
The problem is: **Given K, find a partition of K clusters that optimizes the chosen partitioning criterion**

### K-means

* Given K, select K points as initial centroids
* Repeat
  * Form K clusters by assigning each point to its closest centroid
  * Re-compute the centroids of each cluster
* Convergence criterion is satisfied

Efficienty: O(tKn) where n is the number of objects, K is the number of clusters, t is the number of iterations

* PS: 在给定数据集的情况下，clusters的数量可以通过某种方式确定(X-means)，因此如何更快速的收敛决定了算法的速度

K-means clustering oftern terminates at a local optimal

* PS: So initialization is important, K-means++

**How to determine the best K automatically**

* X-means

Sensitive to noisy data and outliers

* Variations: Using K-medians, K-medoids

Not suitable to discover clusters with non-convex shapes

* Using density-based clustering, kernel K-means

### K-medoids

* Select K points as the initial representative objects
* Repeat
  * Assign each point to the cluster with the closest medoid
  * Randomly select a non-representative object $o_i$
  * Compute the total cost S of swapping the medoid m with $o_i$
  * If S < 0, then swap m with $o_i$ to form the new set of medoids
* Convergence criterion

How to compute cost S

* SSE

Computational complexity: $O(K(n-K)^2)$

CLARA: K-medoids (PAM) on samples, $O(Ks^2 + K(n - K))$, s is the sample size

CLARANS: Randomized re-sampling, ensuring efficiency and quality

### K-medians

每次迭代需要对，cluster中的样本进行排序，得到median

Using L1-norm as distance metric

Cannot handle non-numerical data

### K-Prototype

A mixture of categorial and numerical data

### Kernel K-means

Gaussian radial basis function kernel $K(X_i, X_j) = exp(-||X_i-X_j||^2/2\sigma^2)$

Suppose there are 5 original 2-dimensional points:

|      | x    | y    |
| ---- | ---- | ---- |
| x_1  | 0    | 0    |
| x_2  | 4    | 4    |
| x_3  | -4   | 4    |
| x_4  | -4   | -4   |
| x_5  | 4    | -4   |

If we set $\sigma=4$, in kernel space

|      | $K(x_i, x_1)$ | $K(x_i, x_2)$ | $K(x_i, x_3)$ | $K(x_i, x_4)$ | $K(x_i, x_5)$ |
| ---- | ------------- | ------------- | ------------- | ------------- | ------------- |
| x_1  | 0             | $e^{-1}$      | $e^{-1}$      | $e^{-1}$      | $e^{-1}$      |
| x_2  | $e^{-1}$      | 0             | $e^{-2}$      | $e^{-4}$      | $e^{-2}$      |
| x_3  | $e^{-1}$      | $e^{-2}$      | 0             | $e^{-2}$      | $e^{-4}$      |
| x_4  | $e^{-1}$      | $e^{-4}$      | $e^{-2}$      | 0             | $e^{-2}$      |
| x_5  | $e^{-1}$      | $e^{-2}$      | $e^{-4}$      | $e^{-2}$      | 0             |

也就是说如果有n个数据，就是将数据从2维映射到n维，再利用普通的k-mean来进行聚类