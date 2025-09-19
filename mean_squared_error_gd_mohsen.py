import numpy as np

def compute_mse(y, tx, w):
    """Calculate the loss using either MSE.

    Args:
        y: shape=(N, )
        tx: shape=(N,p)
        w: shape=(p,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # compute loss by MSE
    N = len(y)
    return 1/(2*N) * np.linalg.norm(y - tx @ w)**2

def compute_gradient_mse(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,p)
        w: shape=(p, ). The vector of model parameters.

    Returns:
        An array of shape (p, ) (same shape as w), containing the gradient of the loss at w.
    """
    # compute gradient vector
    N = len(y)
    return -1/N * tx.T @ (y - tx @ w)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """The Gradient Descent (GD) algorithm.

    Args:
        y: shape=(N, )
        tx: shape=(N,p)
        initial_w: shape=(p, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma: a scalar denoting the stepsize

    Returns:
        w: the final estimate of the model parameters, an array of shape (p, )
        loss: the value of the loss (a scalar) for the final w.
    """
    # Initialize parameters
    w = initial_w
    for n_iter in range(max_iters):
        #compute gradient
        grad = compute_gradient_mse(y, tx, w)
        #update w by gradient
        w = w - gamma * grad
        # compute and print current loss
        loss = compute_mse(y, tx, w)
        print(
            "GD iter. {bi}/{ti}: loss={l}".format(
                bi=n_iter, ti=max_iters - 1, l=loss
            )
        )
    return w, loss
