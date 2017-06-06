# Lesson4 Hierarchical Clustering Methods

### Distance

Centroid (中心店)
$$
\hat x_0 = {\sum^n_i\hat x_i \over n}
$$
Radius (所有点到中心点距离的平方的平均的开方)
$$
R = \sqrt{\sum^n_i(\hat x_i - \hat x_0)^2 \over n}
$$
Diameter (类中的n个点，共有n(n-1)个组合，对这些组合的距离做处理)
$$
D = \sqrt{\sum^n_i\sum^n_j(\hat x_i - \hat x_j)^2\over n(n-1)}
$$

### Agglomerative or Divisive Clustering

#### Single link (Nearest Neighbor)

* The similarity between two clusters is the similarity between their most similar members
* **Local similarity-based**: Emphasizing more on close regions, ignoring the overall structure of the cluster
* Capable of clustering **non-elliptical** shaped group of objects
* Sensitive to noise and outliers
* Time complexity is $O(n^2)$

#### Average link (Group Average)

* The average distance between an element in one cluster and an element in the other
* Expensive to compute

#### Complete link (Diameter)

* The similarity between two clusters is the similarity between their most dissimilar members
* Merge two clusters to form one with the smallest **diameter**
* **Nonlocal** in behavior, obtaining compact shaped clusters
* Sensitive to outliers
* Expensive to compute

#### Centroid link (Centroid Similarity)

The distance between the centroids of two clusters

#### Ward's criterion

The increase in the value of the SSE criterion for the clustering obtaiend by merging them into $C_a\bigcup C_b$
$$
W(C_{a\bigcup b}, c_{a\bigcup b}) - W(C, c) = \frac {N_aN_b} {N_a + N_b}d(c_a, c_b)
$$
where, $N_a$ is the **cardinality** of cluster $C_a$, and $c_a$ is the centroid of $C_a$

### BIRCH

Incrementally construct a **CF (Clustering Feature) tree**, a hierarchical data structure for multiphase clustering

* Phase1: Scan DB to build an initial in-memory CF tree
* Phase2: Use an arbitrary clustering algorithms to cluster the leaf nodes of the CF-tree

#### Clustering Feature tree

Clustering feature is the summary of the statistics for a given sub-cluster: the 0th, 1st and 2nd moments of the sub-cluster from the statistic point of view

$CF = (N, LS, SS)$

* N is the number of data points
* LS is the linear sum of N points
* SS is the square sum of N points

|      | x    | y    |
| ---- | ---- | ---- |
| x_1  | 3    | 4    |
| x_2  | 2    | 6    |
| x_3  | 4    | 5    |
| x_4  | 4    | 7    |
| x_5  | 3    | 8    |

$CF = (5, (16, 30), (54, 190))$

There are three types of layers in the tree: root, non-leaf node and leaf node

* The non-leaf nodes store sums of the CFs of their children
* The leaf node store the entry

The algorithm is:

* Find closest leaf entry
* Add point to leaf entry and update CF
* If entry diameter > max_diameter -> split leaf and possibly parents

The algorithm has two parameters:

* Branching factor: Maximum number of children
* Maximum diameter of sub-clusters stored at the leaf nodes

#### Concerns

* Sensitive to insertion order of data points
* Due to the fixed size of leaf nodes, clusters may not be so natural
* Clusters tend to be spherical given the radius and diameter measures

### CURE (Clustering using representitives)

Represent a clustering using a set of well-scattered representative points (the representitive point is same as that in K-medoids), so this algorithm incorporates features of both single link and average link

Shrinking factor $\alpha$ is a hyper-parameter by which the points are shrunk towards the centroid, so the algorithm is more robust to outliers

**But to get representitive points is also time-consuming ??** 

### CHAMELEON

KNN-Graph，需要有权重，可以先不关注