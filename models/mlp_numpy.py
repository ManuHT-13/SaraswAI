"""
Red neuronal MLP implementada en NumPy/CuPy para clasificar instrumentos musicales
a partir de vectores de features extraidos con CNN14
"""

import copy
import cupy as np


def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return (z > 0).astype(float)

def softmax(z):
    # Restamos el maximo por estabilidad numerica
    z_stable = z - np.max(z, axis=1, keepdims=True)
    exp_z    = np.exp(z_stable)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def init_weights(layer_sizes, seed=42):
    np.random.seed(seed)
    thetas = []
    for i in range(len(layer_sizes) - 1):
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        W = np.random.randn(fan_out, fan_in + 1) * np.sqrt(1.0 / fan_in)
        thetas.append(W)
    return thetas


def forward_propagation(thetas, X):
    m     = X.shape[0]
    cache = []
    a = X
    for i, theta in enumerate(thetas):
        a_bias = np.hstack([np.ones((m, 1)), a])
        z      = a_bias @ theta.T
        cache.append((a_bias, z))
        if i == len(thetas) - 1:
            a = softmax(z)
        else:
            a = relu(z)
    return a, cache


def cost(thetas, h, y, lambda_, class_weights=None):
    m = y.shape[0]
    h = np.clip(h, 1e-12, 1 - 1e-12)
    if class_weights is not None:
        weights = np.sum(y * class_weights, axis=1, keepdims=True)
        J = -(1 / m) * np.sum(weights * np.sum(y * np.log(h), axis=1, keepdims=True))
    else:
        J = -(1 / m) * np.sum(y * np.log(h))
    reg = sum(np.sum(theta[:, 1:] ** 2) for theta in thetas)
    J  += (lambda_ / (2 * m)) * reg
    return J


def backprop(thetas, X, y, lambda_, class_weights=None):
    m = X.shape[0]
    h, cache = forward_propagation(thetas, X)
    J        = cost(thetas, h, y, lambda_, class_weights)
    grads    = [None] * len(thetas)

    if class_weights is not None:
        weights = np.sum(y * class_weights, axis=1, keepdims=True)
        delta   = weights * (h - y)
    else:
        delta = h - y

    for i in reversed(range(len(thetas))):
        a_bias, z = cache[i]
        grad = (1 / m) * (delta.T @ a_bias)
        grad[:, 1:] += (lambda_ / m) * thetas[i][:, 1:]
        grads[i] = grad
        if i > 0:
            _, z_prev = cache[i - 1]
            delta = (delta @ thetas[i][:, 1:]) * relu_derivative(z_prev)

    return J, grads


def training(X, y, thetas_ini, alpha, num_iters, lambda_, batch_size=256, class_weights=None, X_val=None, y_val=None):
    thetas = copy.deepcopy(thetas_ini)
    m      = X.shape[0]

    beta1, beta2, eps = 0.9, 0.999, 1e-8
    ms = [np.zeros_like(t) for t in thetas]
    vs = [np.zeros_like(t) for t in thetas]
    t  = 0

    J_history     = []
    J_val_history = []

    for i in range(num_iters):
        idx        = np.random.permutation(m)
        X_s, y_s   = X[idx], y[idx]
        iter_loss  = 0.0
        n_batches  = 0

        for start in range(0, m, batch_size):
            X_batch = X_s[start : start + batch_size]
            y_batch = y_s[start : start + batch_size]

            J, grads = backprop(thetas, X_batch, y_batch, lambda_, class_weights)
            t += 1

            for k in range(len(thetas)):
                ms[k] = beta1 * ms[k] + (1 - beta1) * grads[k]
                vs[k] = beta2 * vs[k] + (1 - beta2) * grads[k]**2
                m_hat = ms[k] / (1 - beta1**t)
                v_hat = vs[k] / (1 - beta2**t)
                thetas[k] -= alpha * m_hat / (np.sqrt(v_hat) + eps)

            iter_loss += J
            n_batches += 1

        J_history.append(iter_loss / n_batches)

        if X_val is not None:
            h_val, _ = forward_propagation(thetas, X_val)
            J_val    = cost(thetas, h_val, y_val, lambda_, class_weights)
            J_val_history.append(float(J_val))

        if (i + 1) % 10 == 0:
            val_str = f"  val_loss={J_val_history[-1]:.4f}" if J_val_history else ""
            print(f"iter {i+1}/{num_iters}  loss={J_history[-1]:.4f}{val_str}")

    return thetas, J_history, J_val_history

def predict(thetas, X):
    h, _ = forward_propagation(thetas, X)
    return np.argmax(h, axis=1)