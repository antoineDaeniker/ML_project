from utils.implementation_utils import penalized_logistic_regression
from utils.implementation_utils import *


def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""

    w = initial_w
    for iter in range(max_iters):
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        w = w - gamma * grad
    return w, loss


def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""

    w = initial_w
    for iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)
            w = w - gamma * grad
    return w, loss


def least_squares(y, tx):
    """calculate the least squares solution."""

    w = np.linalg.solve (tx.T.dot(tx), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    w = np.linalg.solve(tx.T.dot(tx)  + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T.dot(y))
    loss = compute_loss(y, tx, w)
    return w, loss


def logistic_regression(y, tx, initial_w, max_iter, gamma):
    """implement logistic regression."""
    w = initial_w
    for iter in range(max_iter):
        #Using gradient descent
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w = w - gamma * grad
    return w, loss


def reg_logistic_regression(y, tx, lambda_ , initial_w, max_iters, gamma):
    """implement reg_logistic_regression."""
    w = initial_w
    for iter in range(max_iters):
        loss, grad = penalized_logistic_regression(y, tx, w, lambda_) 
        w = w - gamma * grad
    return w, loss


