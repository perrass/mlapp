# Backpropagation: Derivatives

## Vector, Matrix, and Tensor Derivatives

### Gradient: Vector in, scalar out

Suppose $f: \Bbb R^N \to \Bbb R$ takes a vector as input and produces a scalar (**$y$ is a scalar**). The derivate of $f$ at the point $x \in \Bbb R^N$ is now called the gradient.
$$
\nabla_x f(x) = lim_{h\to 0} \frac {f(x+h)-f(x)} {||h||}
$$
Then, we set $y = f(x)$
$$
x \to x + \nabla x => y \to y + \frac {\partial y} {\partial x} \cdot \Delta x
$$
where
$$
\frac {\partial y} {\partial x} = (\frac {\partial y} {\partial x_1}, ..., \frac {\partial y} {\partial x_N})
$$
In particular when multiplying $\frac {\partial y} {\partial x}$ by $\Delta x$ we use the dot product, which combines two vectors to give a scalar.

### Jacobian: Vector in, Vector out

Suppose $f: \Bbb R^N \to \Bbb R^M$ takes a vector as input and produces a vector as output. Then the derivative of $f$ at a point $x$, also called the **Jacobian**, is the $M\times N$ matrix of partial derivatives. 
$$
\frac {\partial y} {\partial x} = \begin{pmatrix}
\frac {\partial y_1} {\partial x_1} & \cdots & \frac {\partial y_1} {\partial x_N} \\
\vdots & \ddots & \vdots \\
\frac {\partial y_M} {\partial x_1} & \cdots & \frac {\partial y_M} {\partial x_N}
\end{pmatrix}
$$
The chain rule can be extended to the vector case using Jacobian matrices. Suppose that $f: \Bbb R^N \to \Bbb R^M$ and $g: \Bbb R^M \to \Bbb R^K$. Let $x \in \Bbb R^N, y \in \Bbb R^M$, and $z \in \Bbb R^K$ with $y=f(x)$ and $z=g(y)$

The chain rule also has the same form as the scalar case:
$$
\frac {\partial z} {\partial x} = \frac {\partial z} {\partial y} \frac {\partial y} {\partial x}
$$

  ### Tensor in, Tensor out

Suppose now that $f: \Bbb R^{N_1\times \cdots N_{D_x}}\to \Bbb R^{M_1 \times \cdots M_{D_x}}$. Then the input to $f$ is $D_x$-dimensional tensor of shape $N_1 \times \cdots N_{D_x}$, and the output of $f$ is $D_y$-dimensional tensor of shape $M_1 \times \cdots M_{D_x}$. Then the derivative $\frac {\partial y} {\partial x}$ is with shape
$$
(M_1 \times \cdots \times M_{D_y}) \times (N_1\times \cdots \times N_{D_x})
$$
\We have seperated the dimensions of $\frac {\partial y} {\partial x}$ into two groups: the first group matches the dimensions of $y$ and the second group matches the dimensions of $x$. With this grouping, we can think of the generalized Jacobian as generialization of a matrix, where each row has the same shape as $y$ and each column has the same shape as $x$.
$$
x \to x + \Delta x => y \to y + \frac {\partial y} {\partial x} \Delta x
$$
where
$$
(\frac {\partial y} {\partial x}\Delta x)_j = \sum_i(\frac {\partial y} {\partial x})_{i, j}(\Delta x)_i = (\frac {\partial y} {\partial x})_{j,:}\cdot \Delta x
$$
In the equation above the term $(\frac {\partial y} {\partial x})_{j, :}$ is the **jth row of the generialized matrix $\frac {\partial y} {\partial x}$, which is a tensor with the same shape of x**.

## How to Compute Derivatives

Computational graph: backpropagation and forward-mode differentiation use a powerful pair of tricks (linearization and dynamic programming) to compute derivatives more efficiently than one might think possible

### Manual

### Numerical Differentiation

### Symbolic Differentiation

### Automatic Differentiation

