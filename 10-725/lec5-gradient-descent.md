# Gradient Descent

## Unconstrained Minimization Problems

Consider **unconstrained, smooth** convex optimization
$$
min_x f(x)
$$
Gradient descent: $x^k = x^{k-1} - t_k\cdot \nabla f(x^{k-1})$

![](assets/gradient.png)

For gradient, it is the tangient line at the **blue point**. However, if we want to increase the speed of convergence, we can add an approximation of second order term (**Hessian**: $\nabla^2 f(x)$), by $\frac 1 t \mathbf I$

Hence, the **red point** is
$$
f(y) \approx f(x) + \nabla f(x)^T(y-x)+\frac 1 {2t} ||y-x||^2_2
$$
This is also the movitation of **Proximal Gradient**

## Gradient Descent & Steepest Gradient Descent

## Gradient Boost

