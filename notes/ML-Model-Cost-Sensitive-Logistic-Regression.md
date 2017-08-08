# ML-Model-Cost-Sensitive-Logistic-Regression

Solver: `scipy.optimize.minimize(method="BFGS")`

Objective Function: `logistic_cost_loss`

```python
def _logistic_cost_loss_i(w, X, y, cost_mat, alpha):
    n_samples = X.shape[0]
    w, c, z = _intercept_dot(w, X)
    y_prob = _sigmoid(z)

    out = cost_loss(y, y_prob, cost_mat) / n_samples
    out += .5 * alpha * np.dot(w, w)
    return out

def _logistic_cost_loss(w, X, y, cost_mat, alpha):
    # alpha is l2 norm
    if w.shape[0] == w.size:
        # Only evaluating one w
        return _logistic_cost_loss_i(w, X, y, cost_mat, alpha)

    else:
        # Evaluating a set of w
        n_w = w.shape[0]
        out = np.zeros(n_w)

        for i in range(n_w):
            out[i] = _logistic_cost_loss_i(w[i], X, y, cost_mat, alpha)

        return out
```

Cost_loss

```python
def cost_loss(y_true, y_pred, cost_mat):
    y_true = column_or_1d(y_true)
    y_true = (y_true == 1).astype(np.float)
    y_pred = column_or_1d(y_pred)
    y_pred = (y_pred == 1).astype(np.float)
    cost = y_true * ((1 - y_pred) * cost_mat[:, 1] + y_pred * cost_mat[:, 2])
    cost += (1 - y_true) * (y_pred * cost_mat[:, 0] + (1 - y_pred) * cost_mat[:, 3])
    return np.sum(cost)
```

$$
Cost(f(S)) = \sum^N_{i=1}[y_i(c_iC_{TP_i} + (1-c_i)C_{FN_i}) + (1-y_i)(c_iC_{FP_i} + (1-c_i)C_{TN_i})]
$$

where $y_i$ is true, and $c_i$ is predicted probability, and the main cost is from $y_i(1-c_i)C_{FN_i}$

Logistic_cost_loss
$$
L(f(S)) = \sum^N_{i=1}[y_i(c_iC_{TP_i} + (1-c_i)C_{FN_i}) + (1-y_i)(c_iC_{FP_i} + (1-c_i)C_{TN_i})] / N + \frac 1 2 \lambda ||\mathbf w ||^2_2
$$
Saving_score
$$
Savings(f(S)) = \frac {Cost(f(S)) - Cost_l(S)} {Cost_l(S)}
$$
Benchmark: Cost(f(S)) = 现行政策的结果