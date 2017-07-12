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

Practically, we set $\Delta = 1.0$ which is safe for all cases. These two parameters $\Delta$ and $\lambda$ are all regularizers, but with the same functionality (rather than L1 and L2). The effect of change of $\Delta$, can be get from changing $\lambda$, because the weights can **shrink or stretch the differences arbitrarily**. Hence, **the only real tradeoff is how large we allow the weights to grow by tuning $\lambda$ **

#### Optimization in primal

In most machine learning materials, SVM is trained by the tricks of **kernels, duals, or the SMO algorithm**. However, in this class, we will always work with the optimization objectives in their **unconstrained primal form**. Many of these objectives are technically not differentiable, but in practice this is not a problem and it is common to use a **subgradient**

#### Other Multiclass SVM formulations



---

## Softmax