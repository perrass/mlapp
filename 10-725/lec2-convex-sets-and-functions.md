# Convext Sets and Functions

## Convex Optimization Problem

$$
\begin{align}
min_{x\in D} & \quad f(x) \\
subject ~ to & \quad g_i(x) \le 0, i = 1,...,m \\
& \quad h_j(x) = 0, j = 1, ..., m
\end{align}
$$

where $f$ and $g_i$ are all convex, and $h_j$ are affine. Any local minimizer is a global minimizer. 

## Convex Sets

#### Defination

$$
x, y\in C => tx + (1-t)y \in C \quad for \ all \ 0\le t\le 1
$$

在$C$上的两点$x,y$，其连线$tx+(1-t)y$中的任意点也在$C$中

#### Examples

* Norm ball: $\{x:  ||x|| \le r\}$, for given norm and radius $r$
* Hyperplane: $\{x:  a^Tx = b\}$, for given a, b. A hyperplane is a affine set
* Halfspace: $\{x:  a^Tx \le b\}$
* Affine: $\{x:  A^Tx = b\}$, for given A, b. A set $C\subseteq \mathbf R^n$ is affine if the line through any two distinct points in C lies in C, i.e., if for any $x_1, x_2 \in C$, and $\theta \in \mathbf R$, we have $\theta x_1 + (1-\theta)x_2 \in C$. Compare the definition of convext sets and affine, the distinction is **convex set is more rigious with $0 \le \theta \le 1$.**
* Polyhedron: $\{x:  A^Tx \le b\}$, the intersection of a finite number of halfspace and hyperplanes. Affine sets(e.g., subspaces, hyperplanes, lines), rays, line segments, and halfspace are all polyhedra.
* Simplex

---

**Ps: the set $\{x:  A^Tx \le b, Cx=d\}$ is also a polyhedron, and why ?**

The formula $Cx=d$ is just a set of constraints, $cx\le d; -cx\le -d$

---

### Cones	

A set C is a convex cone if it is convex and a cone, which means that for any $x_1, x_2 \in C$ and $\theta_1, \theta_2 \ge 0$, we have
$$
\theta_1 x_1 + \theta_2x_2 \in C
$$

#### Examples

* Norm Cone: $\{(x, t): ||x||\le t\}$, for a norm $||\cdot||$
* Normal Cone: given any set $C$ and point $x\in C$, we can define $N_c(x) = \{g:g^Tx \ge g^Ty, ~ for ~ all ~ y\in C\}$. **This is always a convex cone**
* Positive semidefinite cone $\Bbb S^n_+ = \{X\in \Bbb S^n: X \ge 0 \}$, where $X\ge 0$ means that X is positive semidefinite, and $\Bbb S^n$ is the set of $n\times n$ symmetric matrics

### Key properties of convex sets

#### Seperating hyperplane theorem

Suppose $C$ and $D$ are two convex sets that do not intersect, i.e., $C\cap D = \varnothing$. Then there exist $a\neq0$ and $b$ such that $a^Tx \le b$ for all $x\in C$ and $a^Tx \ge b$ for all $x\in D$. In other words, the affine function $a^Tx-b$ is nonpositive on $C$ and nonnegetive on $D$. The hyperplane $\{x | a^T x = b\}$ is called a **seperating hyperplane**

#### Supporting hyperlane theorem

If $a \neq 0$ satisfies $a^Tx \le a^Tx_0$ for all $x\in C$, then the hyperplane $\{x|a^Tx = a^Tx_0\}$ is called a **supporting hyperplace** to $C$ at point $x_0$. The geometric interpretation is that the hyperplane $\{x|a^Tx = a^Tx_0\}$ is tangent to $C$ at $x_0$, and the halfspace $\{x|a^Tx \le a^Tx_0\}$ contains $C$

A boundary point of a convex set has a supporting hyperlane passing through it. Formally, if $C$ is a nonempty convex set, and $x_0 \in bd(C)$, then there exists $a$ such that 
$$
C \subseteq \{x: a^Tx \le a^Tx_0\}
$$

### Operations preserving convexity

**Convexity is preserved under intersection**

#### Affine functions

**A function $f: \mathbf R^n \to \mathbf R^m$ is *affine* if it is a sum of a linear function and a constant**, i.e., if it is has the form $f(x) = Ax+b$, where $A\in \mathbf R^{m\times n}$ and $b \in \mathbf R^m$. Suppose $S \subseteq \mathbf R^n$ is convex and $f: \mathbf R^n \to \mathbf R^m$ is an affine function. Then the image of $S$ under $f$, $f(S) = \{f(x)|x\in S\}$, is convex. 

---

Example: Scaling and Translation

If $S \subseteq \mathbf R^n$ is convex, then the sets $\alpha S$ and $S+\alpha$ are convex.

---

Similarly, if $f: \mathbf R^k \to \mathbf R^n$ is an affine function, the inverse image of $S$ under $f$, $f^{-1}(S) = \{x|f(x)\in S\}$ is convex.

---

Example: **Linear Matrix Inquality**

Given $A_1, ..., A_k, B \in \mathbf S^n$, a linear matrix inequality is of the form 

$$
x_1A_1 + x_2A_2 + ...+x_kA_k \le B
$$
for a variable $x \in \Bbb R^k$. Then, **the set $C$ of points x that satisfy the above inequality is convex**

Solution:

It is the **inverse image of the positive semidefinite cone** under the affine function $f: \mathbf R^k \to \mathbf S^n$ given by $f(x) = B - A(x)$. 

Or, let $f: \mathbf R^k \to \mathbf S^n$, $f(x) = B - \sum^k_{i=1}x_iA_i$.  $C = f^{-1}(\mathbf S^n_+)$

---

#### Perspective and linear-fractional functions

## Convex Functions

#### Definition

$f:\Bbb R^n ->\Bbb R$ such that $dom(f) \subseteq \Bbb R^n$ **convex**, and 
$$
f(tx  + (1-t)y) \le tf(x) + (1-t)f(y) \quad  0 \le t \le 1
$$
and all $x, y \in dom(f)$

A function $f$ is **strictly convex** if strict inequality holds the former formular whenever $x \neq y$ and $0< t < 1$

#### Examples

* Exponential:  $e^{ax}$
* Powers: $x^a$ where x > 0 and $a \ge 1$ or $a\le 0$
* Powers of absolute value $|x|^p$, for $p\ge 1$
* Logarithm: $-log x$ on $\Bbb R_{++}$
* Negative entropy: $xlogx$ on $\Bbb R_{++}$
* **Affine function**: $a^Tx+b$ is both convex and concave
* **Quadratic function**: $\frac 1 2 x^TQx + b^Tx + c$ is convex provided that $Q \ge 0$ (positive semidefinite) due to $\nabla^2f(x) = Q$
* **Least squares loss**: $||y-Ax||^2_2$, since $A^TA$ is always positive semidefinite
* **Norms**. Every norm on $\Bbb R^n$ is convex
* **Max**. on $\Bbb R^n$
* **Quadratic-over-linear-function**: $f(x,y) = x^2/y$, with $dom ~f  = \Bbb R \times \Bbb R_{++} = \{(x, y) \in \Bbb R^2 | y > 0\}$ is convex
* **Log-sum-exp**: $f(x) = log(e^{x_1} + ... + e^{x_n})$
* **Geometric mean**: $f(x) = (\prod^n_{i=1}x_i)^{1/n}$ is **concave** on $dom ~ f = \Bbb R^n_{++}$

### Key properties of convex functions

#### First-order conditions

Suppose $f$ is differentiable. Then $f$ is convex if and only if $dom ~ f$ is convex and 
$$
f(y) \ge f(x) + \nabla f(x)^T(y-x)
$$
holds for all $x, y \in dom ~ f$.

This states that **for a convex function, the first order Taylor approximation is in fact a global underestimator of the function**, and **vice versa**. This means from **local information** about a convex function, we can derive **global information**. Furthermore, if $\nabla f(x) = 0$, then for all $y\in dom ~ f$, $f(y) \ge f(x)$, i.e., x is a global minimizer of the function $f$.

#### Second-order conditions

Suppose $f$ is twice differentiable, that is, its **Hessian** $\nabla^2f$ exists at each point in $dom ~ f$, which is open. Then $f$ is convex if and only if $dom ~ f$ is convex and its Hessian is positive semidefinite: for all $x \in dom ~ f$

---

**Proofs of convexity from former examples**

---

#### Sublevel sets

The a-sublevel set of a function $f: \Bbb R^n \to \Bbb R$ is defined as
$$
C_\alpha = \{x \in dom ~ f | f(x)\le \alpha \}
$$
Sublevel sets of a convex function are convex.

#### Epigraph

#### Jensen's inequality

$$
f(tx  + (1-t)y) \le tf(x) + (1-t)f(y)
$$

is sometimes called **Jensen's inequality**. It is easily extended to convex combinations of more than two points: If $f$ is convex, $x_1, ..., x_k \in dom ~ f$, and $\theta_1, ..., \theta_k \ge 0$ with $\theta_1 + ... + \theta_k = 1$, then
$$
f(\theta_1x_1 + ... + \theta_kx_k) \le \theta_1f(x_1) + ... + \theta_kf(x_k)
$$
Furthermore, If $x$ is a random variable such that $x\in dom ~ f$ with probability one, and $f$ is convex, then we have
$$
f(\mathbf Ex) \le \mathbf Ef(x)
$$

### Operations preserving convexity