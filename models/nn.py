import numpy as np
import matplotlib.pyplot as plt
import utils
import argparse
from scipy.io import loadmat
import copy


def cost(theta1, theta2, X, y, h, lambda_):
    """
    Compute cost for 2-layer neural network. 

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    """

    m = X.shape[0]

    # Compute cost function for current theta values with the regularization term
    reg = (lambda_ / (2*m)) * (np.sum(theta1[:, 1:]**2) + np.sum(theta2[:, 1:]**2))
    J = -(1/m) * np.sum(y*np.log(h) + (1-y)*np.log(1-h)) + reg

    return J


def forward_propagation(theta1, theta2, X):
    """
    Gives Neural Network prediction for a given theta values.

    Parameters
    ----------
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    Returns
    -------
    h : array_like
        Predictions of the Neural Network for 

    """

    m = X.shape[0] 

    # Add a column of ones to the first layer input for bias
    a1 = np.hstack([np.ones((m, 1)), X])

    # Compute first layer output and add a column of ones
    z2 = a1 @ theta1.T
    a2 = utils.sig(z2)
    a2 = np.hstack([np.ones((m, 1)), a2])

    # Compute nn output
    z3 = a2 @ theta2.T
    h = utils.sig(z3)

    return h, a1, z2, a2, z3


def backprop(theta1, theta2, X, y, lambda_):
    """
    Compute cost and gradient for 2-layer neural network. 

    Parameters                                          
    ----------                                          
    theta1 : array_like
        Weights for the first layer in the neural network.
        It has shape (2nd hidden layer size x input size + 1)

    theta2: array_like
        Weights for the second layer in the neural network. 
        It has shape (output layer size x 2nd hidden layer size + 1)

    X : array_like
        The inputs having shape (number of examples x number of dimensions).

    y : array_like
        1-hot encoding of labels for the input, having shape 
        (number of examples x number of labels).

    lambda_ : float
        The regularization parameter. 

    Returns
    -------
    J : float
        The computed value for the cost function. 

    grad1 : array_like
        Gradient of the cost function with respect to weights
        for the first layer in the neural network, theta1.
        It has shape (2nd hidden layer size x input size + 1)

    grad2 : array_like
        Gradient of the cost function with respect to weights
        for the second layer in the neural network, theta2.
        It has shape (output layer size x 2nd hidden layer size + 1)

    """

    m = X.shape[0]

    # Get prediction of the nn and layer info 
    h, a1, z2, a2, z3 = forward_propagation(theta1, theta2, X)

    # Compute cost for current theta values
    J = cost(theta1, theta2, X, y, h, lambda_)

    # Compute error of the last layer
    delta3 = h - y

    # Propagate error to the previous layer
    delta2 = (delta3 @ theta2[:, 1:]) * utils.sig_derivative(z2)

    # Compute gradients for both theta vectors
    grad2 = (1/m) * (delta3.T @ a2)
    grad1 = (1/m) * (delta2.T @ a1)

    # Regularize non-bias components 
    grad2[:, 1:] += (lambda_/m) * theta2[:, 1:]
    grad1[:, 1:] += (lambda_/m) * theta1[:, 1:]

    return (J, grad1, grad2)


def training(X, y, theta1_ini, theta2_ini, backprop, alpha, num_iters, lambda_=None):
    theta1 = copy.deepcopy(theta1_ini)
    theta2 = copy.deepcopy(theta2_ini)
    J_history = []

    for iter in range(num_iters):
      J, grad1, grad2 = backprop(theta1, theta2, X, y, lambda_)

      theta1 -= alpha*grad1
      theta2 -= alpha*grad2

      J_history.append(J)

    return theta1, theta2, J_history


def predict(theta1, theta2, X):
    h, _, _, _, _ = forward_propagation(theta1, theta2, X)
    return np.argmax(h, axis=1)


def main_train():
    # Load data
    data = loadmat('data/ex3data1.mat', squeeze_me=True)
    y = data['y']
    X = data['X']

    # Suffle data
    indexes = np.random.permutation(X.shape[0])
    X = X[indexes]
    y = y[indexes]

    # Code y labels to a vector
    y_coded = np.eye(10)[y]
 
    # Slice the data
    X_train = X[:4400,:]
    X_test = X[4400:,:]
    y_train = y_coded[:4400,:]
    y_test = y[4400:]

    # Initialize parameters
    input_size = X.shape[1]
    hidden_size = 40
    output_size = y_coded.shape[1]

    epsilon = 0.12
    theta1_ini = np.random.rand(hidden_size, input_size + 1) * 2 * epsilon - epsilon
    theta2_ini = np.random.rand(output_size, hidden_size + 1) * 2 * epsilon - epsilon

    alpha = 1
    lambda_ = 1
    num_iters = 1000

    # Start training
    theta1, theta2, J_history = training(X_train, y_train, theta1_ini, theta2_ini, backprop,
                                         alpha, num_iters, lambda_)
    
    # Graph cost evolution
    xs = np.linspace(0, num_iters, num_iters)
    plt.plot(xs, J_history)

    plt.show()

    # Test model accuracy
    y_pred = predict(theta1, theta2, X_test)
    accuracy = np.mean(y_pred == y_test) * 100
    print(f"Accuracy: {accuracy:.2f}%")

    # Save parameters
    np.savez('thetas.npz', theta1=theta1, theta2=theta2)


def main_predict(image_route):
    # Load data
    data = loadmat('data/ex3data1.mat', squeeze_me=True)
    y = data['y']
    X = data['X']

    # Load parameters
    data = np.load('thetas.npz')
    theta1 = data['theta1']
    theta2 = data['theta2']

    # Get normalization params for image info
    mean = X.mean()
    std = X.std()

    # Load image
    img = plt.imread(image_route) 

    # Convert to gray scale if it is RGB
    if img.ndim == 3:
        img = img.mean(axis=2) 

    # Invert background color
    img = 1 - img

    # Flat and normalize
    img_vector = img.T.flatten()
    img_vector = (img_vector - X.mean()) / X.std()

    # Display image
    utils.displayImage(img_vector)
    plt.show()

    # Get NN prediction
    pred = predict(theta1, theta2, img_vector.reshape(1, -1))
    print(f"The model sees: {pred[0]}")


def main():
    # Parse commands 
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], required=True)
    parser.add_argument('--image', type=str, help='image route')
    args = parser.parse_args()

    if args.mode == 'train':
        main_train()
    elif args.mode == 'predict' and not args.image is None:
        main_predict(args.image)

main()