"""
Red neuronal MLP implementada en NumPy/CuPy

* Uso Cupy porque corre en GPU y asi hago el entreno mas rapido, opcional.
Si se activa aqui activar tambien en train_mlp y en cross_validation

"""

USE_GPU = False

# Parametro de dropout (porcentaje de neuronas aleatorias que apagamos en cada propagacion)
DROPOUT_RATE = 0.3

# Los importamos con ese nombre "Np" para que Cupy NUNCA colisione con otros imports numpy
if USE_GPU:
    import cupy as Np
    Np.cuda.Device(0).use()
else:
    import numpy as Np

import copy

RANDOM_SEED = 429


def relu(z):
    """
    Como funcion de activacion he decidido usar ReLu en lugar de Sigmoid por
    la alta dimensionalidad de los vectores de featurings y su gran magnitud
    """
    return Np.maximum(0, z)

def relu_derivative(z):
    """
    Derivada de la funcion de activacion
    """
    return (z > 0).astype(float)

def softmax(z):
    """
    Funcion de activacion de la capa de salida normalizando las
    salidas como probabilidades que suman 1
    """
    # Restamos el maximo para estabilidad numerica porque si no da problemas
    z_stable = z - Np.max(z, axis=1, keepdims=True)
    exp_z = Np.exp(z_stable)
    return exp_z / Np.sum(exp_z, axis=1, keepdims=True)


def init_weights(layer_sizes, seed=RANDOM_SEED):
    """
    Inicializacion de pesos con He normal, adecuada para ReLU:
    escala por sqrt(2/fan_in) para mantener la varianza estable
    en redes profundas con activaciones ReLU
    """
    Np.random.seed(seed)
    thetas = []
    for i in range(len(layer_sizes) - 1):
        fan_in  = layer_sizes[i]
        fan_out = layer_sizes[i + 1]
        std = Np.sqrt(2.0 / fan_in)
        W = Np.random.randn(fan_out, fan_in + 1) * std
        thetas.append(W)
    return thetas


def forward_propagation(thetas, X, training=False, dropout_rate=DROPOUT_RATE):
    """
    Propagacion hacia adelante calculando la salida de cada capa y guardando
    en bp_ret la a_bias, salida z y mascara de dropout de cada capa para
    pasarsela a la propagacion hacia atras despues. Si training esta a True entonces
    aplicamos dropout apagando neuronas aleatorias de las capas ocultas para compensar
    el sobreajuste
    """
    m = X.shape[0]
    bp_ret = []
    a = X
    for i, theta in enumerate(thetas):
        # Columna de unos para la bias
        a_bias = Np.hstack([Np.ones((m, 1)), a])
        z = a_bias @ theta.T
 
        is_last = (i == len(thetas) - 1)
        if is_last:
            a = softmax(z)
            mask = None
        else:
            a = relu(z)
            # Aplicamos el dropout si estamos entrenando y guardamos la mascara con las neuronas desactivadas
            if training and dropout_rate > 0.0:
                mask = (Np.random.rand(*a.shape) > dropout_rate).astype(a.dtype)
                a = a * mask / (1.0 - dropout_rate) 
            else:
                mask = None
 
        bp_ret.append((a_bias, z, mask))
    return a, bp_ret


def cost(thetas, h, y, lambda_, class_weights=None):
    """
    Calculamos la funcion de coste actual con regularizacion y opcionalmente
    pesos de cada clase penalizando mas los errores en clase minoritarias
    para compensar el desbalanceo del dataset
    """
    m = y.shape[0]
    h = Np.clip(h, 1e-12, 1 - 1e-12)

    if class_weights is not None:
        weights = Np.sum(y * class_weights, axis=1, keepdims=True)
        J = -(1 / m) * Np.sum(weights * Np.sum(y * Np.log(h), axis=1, keepdims=True))
    else:
        J = -(1 / m) * Np.sum(y * Np.log(h))

    reg = sum(Np.sum(theta[:, 1:] ** 2) for theta in thetas)
    J += (lambda_ / (2 * m)) * reg
    return J


def backprop(thetas, X, y, lambda_, class_weights=None):
    """
    Calculamos el gradiente de la funcion de coste respecto a cada peso mediante
    propagacion hacia atras. Primero hacemos la prop. hacia delante y luego propagamos
    el error hacia atras capa a capa respetando las mismas mascaras para
    las neuronas desactivadas por el dropout
    """
    m = X.shape[0]
    h, cache = forward_propagation(thetas, X, training=True)
    J = cost(thetas, h, y, lambda_, class_weights)
    grads = [None] * len(thetas)
 
    # Si tenemos la opcion de pesos por clase avanzamos mas por las clases minoritarias
    if class_weights is not None:
        weights = Np.sum(y * class_weights, axis=1, keepdims=True)
        delta = weights * (h - y)
    else:
        delta = h - y
 
    for i in reversed(range(len(thetas))):
        a_bias, z, mask = cache[i]
        grad = (1 / m) * (delta.T @ a_bias)
        grad[:, 1:] += (lambda_ / m) * thetas[i][:, 1:]
        grads[i] = grad
        if i > 0:
            _, z_prev, mask_prev = cache[i - 1]
            delta = (delta @ thetas[i][:, 1:]) * relu_derivative(z_prev)
            # Propagamos el error solo por las neuronas que no fueron apagadas
            if mask_prev is not None:
                delta = delta * mask_prev / (1.0 - DROPOUT_RATE)
 
    return J, grads


def training(X, y, thetas_ini, alpha, num_iters, lambda_, batch_size=256, class_weights=None, X_val=None, y_val=None):
    """
    Entrenamos la red con el metodo por batches que vimos en la practica 5
    y ademas un Adam muy simple para ir adaptando alpha y converger mas rapidamente
    """
    thetas = copy.deepcopy(thetas_ini)
    m = X.shape[0]

    # Hiperparametros del Adam, beta1 controla el momentum y beta2 la velocidad
    # (cuanto recordamos la direccion del gradiente anterior y cuanto su magnitud)
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    # Momento (media movil del gradiente, direccion) para cada peso
    ms = [Np.zeros_like(t) for t in thetas]
    # Velocidad (media movil al cuadrado, magnitud) para cada peso
    vs = [Np.zeros_like(t) for t in thetas]
    # Contador de pasos 
    t = 0

    J_history = []
    J_val_history = []

    # Para cada iteracion
    for i in range(num_iters):

        t = i + 1

        # Mezclamos los valores en cada episodio para que el orden del dataset
        # no sea un factor en el aprendizaje
        idx = Np.random.permutation(m)
        X_s, y_s = X[idx], y[idx]
        iter_loss = 0.0
        n_batches = 0
 
        for start in range(0, m, batch_size):
            X_batch = X_s[start : start + batch_size]
            y_batch = y_s[start : start + batch_size]

            # Calculamos funcion de coste, y gradientes para cada peso
            J, grads = backprop(thetas, X_batch, y_batch, lambda_, class_weights)

            for k in range(len(thetas)):
                # Calculamos los momentos para cada peso
                ms[k] = beta1 * ms[k] + (1 - beta1) * grads[k]
                vs[k] = beta2 * vs[k] + (1 - beta2) * grads[k]**2
                # Corregimos el sesgo 
                m_hat = ms[k] / (1 - beta1**t)
                v_hat = vs[k] / (1 - beta2**t)

                # Ajustamos los pesos teniendo en cuenta los momentos, de manera que los que reciben
                # un gradiente grande aprenden menos y viceversa
                thetas[k] -= alpha * m_hat / (Np.sqrt(v_hat) + eps)

            iter_loss += J
            n_batches += 1

        J_history.append(iter_loss / n_batches)

        # Calculamos la funcion de coste para el split de valicacion (hemos metido directamente test)
        # de esta manera podemos ver en que punto el modelo empieza a sobreajustarse
        if X_val is not None:
            h_val, _ = forward_propagation(thetas, X_val)
            J_val = cost(thetas, h_val, y_val, lambda_, class_weights)
            J_val_history.append(float(J_val))

        # Imprimimos el progreso cada 10 episodios
        if (i + 1) % 10 == 0:
            val_str = f"  val_loss={J_val_history[-1]:.4f}" if J_val_history else ""
            print(f"iter {i+1}/{num_iters}  loss={J_history[-1]:.4f}{val_str}")

    return thetas, J_history, J_val_history

def predict(thetas, X):
    """
    Devuelve la clase predicha para cada muestra como el indice de mayor
    probabilidad en la salida del softmax
    """
    h, _ = forward_propagation(thetas, X)
    return Np.argmax(h, axis=1)