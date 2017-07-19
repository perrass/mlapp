# Lec3 Optimization Basics

## Optimization

### Basics

$$
\begin{align}
minimize & \quad f_0(x)\\
subject ~ to & \quad f_i(x) \le 0, & \quad i = 1, ..., m \\
& \quad h_i(x) = 0, & \quad i = 1, ..., p
\end{align}
$$

* $f_0(x)$ is objective function
* $f_i(x) \le 0$ are inequality functions and $h_i(x) = 0$ are enquality functions
* The set of points, $D = \bigcap_{i=0}^m dom~ f_i \cap \bigcap_{i=1}^p dom ~ h_i$, is called the **domain** of the optimization. A point $x\in D$ is **feasible** if it satisfies the constraints

### Equivalent problems

#### Change of variables

Suppose $\phi: \Bbb R^n \to \Bbb R^n$ is **one to one**, with image covering the problem domain $D$. We define functions $\tilde f_i(z) = f_i(\phi(z))$ and $\tilde h_i(z) = h_i(\phi(z))$. Then the problem is transformed to **meaning that the two problems are equivalent**
$$
\begin{align}
minimize & \quad \tilde f_0(z)\\
subject to & \quad \tilde f_i(z) \le 0, & \quad i = 1, ..., m \\
& \quad \tilde h_i(z) = 0, & \quad i = 1, ..., p
\end{align}
$$

#### Transformation of objective and constraint functions

Suppose that $\psi_0: \Bbb R \to \Bbb R$ is **monotone increasing**, $\psi_1, ...\psi_m: \Bbb R \to \Bbb R$ satisfy $\psi_i(u) \le 0$ if and only if $u \le 0$, and $\psi_{m+1}, ... \psi_{m+p}: \Bbb R \to \Bbb R$ satisfy $\psi_i(u) = 0$ if and only if $u = 0$. We define functions $\tilde f_i(x) = \psi_i(f_i(x)), i=0,...,m$ and $\tilde h_i(x) = \psi_{m+i}(h_i(x)), i=1,...,p$
$$
\begin{align}
minimize & \quad \tilde f_0(z)\\
subject to & \quad \tilde f_i(z) \le 0, & \quad i = 1, ..., m \\
& \quad \tilde h_i(z) = 0, & \quad i = 1, ..., p
\end{align}
$$
The two problems are also equivalent. This is used to reveal the hidden convexity of a problem

#### Slack variables

One simple **transformation** is based on the observation that $f_i(x)\le 0$ if and only if there is an $s_i \ge 0$ that satisfies $f_i(x) + s_i = 0$. Using this transformation we  obtain the problem
$$
\begin{align}
minimize & \quad f_0(x)\\
subject ~ to & \quad s_i \ge 0 & \quad i = 1, ..., m \\ 
& \quad f_i(x) + s_i =  0,& \quad i = 1, ..., m \\
& \quad h_i(x) = 0, & \quad i = 1, ..., p
\end{align}
$$
where the variable $x \in \Bbb R^n$ and $s \in \Bbb R^m$. This problem has $n+m$ variables, $m$ inequality constraints, and $m+p$ equality constraints. The new variable $s_i$ is called the **slack variable** associated with the original inequality constraint $f_i(x) \le 0$.

#### Eliminating equality constraints

#### Eliminating linear equality constraints

#### Introducing equality constraints

## Convex Optimization

$$
\begin{align}
minimize & \quad f_0(x)\\
subject ~ to & \quad f_i(x) \le 0, & \quad i = 1, ..., m \\
& \quad h_i(x) = 0, & \quad i = 1, ..., p
\end{align}
$$

For convex optimization, where are three additional requirements:

1. the objective function must be convex
2. the inequality constraint functions must be convex
3. the equality constranit functions $h_i(x) = a_i^Tx - b$ must be affine

And, there are two important properties

1. The feasible set of a convex optimization problem is convex
2. **If the objective is strictly convex, then the optimal set contains at most one point**
3. Local minima are global minima

**Proof of property one**

Assume $x, y$ are solutions, let $0\le t \le 1$, and exist a convex set $D$

* $tx + (1-t)y \in D$
* Due to convexity, $g_i(tx + (1-t)y) \le tg_i(x) + (1-t)g_i(y) \le 0$
* $A(tx + (1-t)y) = tAx + (1-t)Ay = b$
* $f(tx+(1-t)y)\le tf(x) + (1-t)f(y)$

Therefor $tx+(1-t)y$ is also a solution.

**Proof of property three**

Suppose that x is locally optimal for a convex optimization porblem, i.e., $x$ is feasible and 
$$
f_0(x) = inf\{f_0(z)|z ~ feasible, ||z-x||_2\le R\}
$$
for some $R>0$. Now suppose that $x$ is not global optimal, i.e., there is a feasible $y$ such that $f_0(y)<f_0(x)$. Evidently $||y-x||_x > R$, since otherwise $f_0(x)\le f_0(y)$. Consider the point $z$ given by 
$$
z = (1-\theta) x + \theta y, \quad \theta = \frac R {2||y-x||_2}
$$
Then we have $||z-x||_2 = R/2 < R$, and by convexity of the feasible set, $z$ is feasible. By convexity of $f_0$ we have
$$
f_0(z) \le (1-\theta)f_0(x) + \theta f_0(y) < f_0(x)
$$
which is **contradicts**. Hence, local minima are global minima

### An optimality criterion for differentiable function

Suppose that the objective $f_0$ in a convex optimization problem is differentiable, so that for $x, y \in dom ~ f_0$,
$$
f_0(y) \ge f_0(x) + \nabla f_0(x)^T(y-x)
$$
Let $X$ denote the feasible set, then $x$ is optimal if and only if $x\in X$ and 
$$
\nabla f_0(x)(y-x)\ge 0 \quad for ~ all ~ y\in X
$$
Geometrically, If $\nabla f_0(x) \neq 0$, it means that $-\nabla f_0(x)$ defines a supporting hyperplane to the feasible set at $x$

### Equivalent convex problems

## Machine Learning Related

### Vector optimization

#### Ridge

#### Lasso

### Partial optimization

#### SVM

Given $y\in \{-1, 1\}^n, X \in \Bbb R^{n\times p}$ with rows $x_1, ..., x_n$, consider the SVM problem:
$$
\begin{align}
minimize & \quad \frac 1 2 ||\beta||^2_2+C\sum^n_{i=1}\xi_i\\
subject ~ to & \quad \xi_i \ge 0 & \quad i = 1, ..., n \\
& \quad y_i(x_i^T\beta+\beta_0)\ge 1 - \xi_i, & \quad i = 1, ..., n
\end{align}
$$
**Partial optimization**: 

Suppose $x=(x_1, x_2)\in \Bbb R^{n_1+n_2}$, then
$$
\begin{align}
minimize & \quad f(x_1, x_2)\\
subject ~ to & \quad g_1(x_1) \le 0 \\
& \quad g_2(x_2)\le 0
\end{align}
$$
is equivalent to 
$$
\begin{align}
minimize & \quad \tilde f(x_1)\\
subject ~ to & \quad g_1(x_1) \le 0 \\
\end{align}
$$
where $\tilde f(x_1) = min\{f(x_1, x_2): g_2(x_2)\le 0\}$. **If the first problem is convex, the second problem is also convex** due to the feasible set of the first problem is convex.

Using partial optimization and hinge loss, we can get the **hinge form of SVMs**
$$
min_{\beta, \beta_0}\qquad \frac 1 2 ||\beta||^2_2+C\sum^n_{i=1}[1-y_i(x_i^T\beta+\beta_0)]_+
$$
**Questions:**

**Is this convex?**: Yes, it is clear that there is a quadratic component and a linear component for object function.

**Is the solution $\{\beta, \beta_0, \xi\}$ unique?**: No, the problem is not **strict convex** due to the linear component, hence the solution is not unique

**Is the $\beta$ part unique?**: Yes, this means the weights are unique.







### 

