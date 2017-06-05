# Lesson4 Hierarchical Clustering Methods

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
