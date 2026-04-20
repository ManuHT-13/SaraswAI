import matplotlib.pyplot as plt
import numpy as np


###########################################################################
# data display
#
def displayData(X):
    num_plots = int(np.size(X, 0)**.5)
    fig, ax = plt.subplots(num_plots, num_plots, sharex=True, sharey=True)
    plt.subplots_adjust(left=0, wspace=0, hspace=0)
    img_num = 0
    for i in range(num_plots):
        for j in range(num_plots):
            # Convert column vector into 20x20 pixel matrix
            # transpose
            img = X[img_num, :].reshape(20, 20).T
            ax[i][j].imshow(img, cmap='Greys')
            ax[i][j].set_axis_off()
            img_num += 1

    return (fig, ax)


def displayImage(im):
    fig2, ax2 = plt.subplots()
    image = im.reshape(20, 20).T
    ax2.imshow(image, cmap='gray')
    return (fig2, ax2)


###########################################################################
# gradient checking
#
def debugInitializeWeights(fan_in, fan_out):
    """
    Initializes the weights of a layer with fan_in incoming connections and
    fan_out outgoing connections using a fixed set of values.
    """

    W = np.sin(np.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    W = W.reshape(fan_out, 1+fan_in, order='F')
    return W


def computeNumericalGradient(J, Theta1, Theta2):
    """
    Computes the gradient of J around theta using finite differences and
    yields a numerical estimate of the gradient.
    """

    theta = np.append(Theta1, Theta2).reshape(-1)

    numgrad = np.zeros_like(theta)
    perturb = np.zeros_like(theta)
    tol = 1e-4

    for p in range(len(theta)):
        # Set perturbation vector
        perturb[p] = tol
        loss1 = J(theta - perturb)
        loss2 = J(theta + perturb)

        # Compute numerical gradient
        numgrad[p] = (loss2 - loss1) / (2 * tol)
        perturb[p] = 0

    return numgrad


def checkNNGradients(costNN, reg_param=0):
    """
    Creates a small neural network to check the backpropagation gradients. It will output the
    analytical gradients produced by your backprop code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result in
    very similar values.
    Parameters
    ----------
    nnCostFunction : func
        A reference to the cost function implemented by the student.
    reg_param : float (optional)
        The regularization parameter value.
    """

    # Set up small NN
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    # Generate some random test data
    Theta1 = debugInitializeWeights(input_layer_size, hidden_layer_size)
    Theta2 = debugInitializeWeights(hidden_layer_size, num_labels)

    # Reusing debugInitializeWeights to get random X
    X = debugInitializeWeights(input_layer_size - 1, m)

    # Set each element of y to be in [0,num_labels]
    y = [(i % num_labels) for i in range(m)]

    ys = np.zeros((m, num_labels))
    for i in range(m):
        ys[i, y[i]] = 1

    # Unroll parameters
    # nn_params = np.append(Theta1, Theta2).reshape(-1)

    # Compute Gradient
    cost, grad1, grad2 = costNN(Theta1, Theta2,
                                X, ys, reg_param)
    grad = np.concatenate((np.ravel(grad1), np.ravel(grad2)))

    def reduced_cost_func(p):
        """ Cheaply decorated nnCostFunction """
        Theta1 = np.reshape(
            p[:hidden_layer_size * (input_layer_size + 1)],
            (hidden_layer_size, (input_layer_size + 1)))
        Theta2 = np.reshape(
            p[hidden_layer_size * (input_layer_size + 1):],
            (num_labels, (hidden_layer_size + 1)))
        return costNN(Theta1, Theta2,
                      X, ys, reg_param)[0]

    numgrad = computeNumericalGradient(reduced_cost_func, Theta1, Theta2)

    # Check two gradients
    # np.testing.assert_almost_equal(grad, numgrad)

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = np.linalg.norm(numgrad - grad)/np.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)


def sig(z):
    return 1/(1+np.exp(-z))


def sig_derivative(z):
    sigmoid = sig(z)
    return sigmoid * (1 - sigmoid)
