import numpy as np
import matplotlib.pyplot as plt

def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    
    return 1/2 * np.mean((y - tx @ w)**2)  #MSE
    #return np.mean(abs(y - tx @ w)) #MAE


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    return -tx.T @ (y - tx @ w) / len(y)


def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        grad = compute_gradient(y, tx, w)
        loss = compute_loss(y, tx, w)
        
        w = w - gamma * grad
        
        # store w and loss
        ws.append(w)
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses[len(losses) - 1], ws[len(ws) - 1]


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]


def stochastic_gradient_descent(y, tx, initial_w, batch_size, max_iters, gamma):
    """Stochastic gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        for minibatch_y, minibatch_tx in batch_iter(y, tx, batch_size):
        
            grad = compute_gradient(minibatch_y, minibatch_tx, w)
            loss = compute_loss(minibatch_y, minibatch_tx, w)

            w = w - gamma * grad

            # store w and loss
            ws.append(w)
            losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, normgrad={normgrad}".format(bi=n_iter, ti=max_iters - 1, l=loss, normgrad=np.linalg.norm(grad)), end='\r')
    return losses, ws[len(ws) - 1]


def least_squares(y, tx):
    """calculate the least squares solution."""

    w = np.linalg.solve (tx.T.dot(tx), tx.T.dot(y))
    return w, compute_loss(y, tx, w)


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    poly = np.ones((len(x), 1))
    for deg in range(1, degree+1):
        poly = np.c_[poly, np.power(x, deg)]
    return poly
#def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    
#    return np.array([x**j for j in range(degree)])


def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    
    w = np.linalg.solve(tx.T.dot(tx)  + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T.dot(y))
    return w, compute_loss(y, tx, w)


def ridge_regression_demo(x, y, degree, ratio, seed):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 0, 15)

    x_tr, y_tr, x_te, y_te = split_data(x, y, ratio, seed=seed)
    
    poly_data_tr = build_poly(x_tr, degree)
    poly_data_te = build_poly(x_te, degree)
    
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        
        w, l = ridge_regression(y_tr, poly_data_tr, lambda_)
        
        rmse_tr.append(np.sqrt(2 * compute_loss(y_tr, poly_data_tr, w)))
        rmse_te.append(np.sqrt(2 * compute_loss(y_te, poly_data_te, w)))
        print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               p=ratio, d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
        
    # Plot the obtained results
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)
    
    
def ridge_regression_demo_2(x, y, degree, ratio, seed):
    """ridge regression demo."""
    # define parameter
    lambdas = np.logspace(-5, 0, 15)

    x_tr, y_tr, x_te, y_te = split_data(x, y, ratio, seed=seed)
    print(x_tr.shape, y_tr.shape)
    
    #poly_data_tr = build_poly(x_tr, degree)
    #poly_data_te = build_poly(x_te, degree)
    
    rmse_tr = []
    rmse_te = []
    for ind, lambda_ in enumerate(lambdas):
        
        w, l = ridge_regression(y_tr, x_tr, lambda_)
        
        rmse_tr.append(np.sqrt(2 * compute_loss(y_tr, x_tr, w)))
        rmse_te.append(np.sqrt(2 * compute_loss(y_te, x_te, w)))
        print("proportion={p}, degree={d}, lambda={l:.3f}, Training RMSE={tr:.3f}, Testing RMSE={te:.3f}".format(
               p=ratio, d=degree, l=lambda_, tr=rmse_tr[ind], te=rmse_te[ind]))
        
    # Plot the obtained results
    plot_train_test(rmse_tr, rmse_te, lambdas, degree)


def split_data(x, y, ratio, seed=1):
    """
    split the dataset based on the split ratio. If ratio is 0.8 
    you will have 80% of your data set dedicated to training 
    and the rest dedicated to testing
    """
    # set seed
    np.random.seed(seed)
    
    indexes = np.random.permutation(len(y))
    index_split = int(np.floor(ratio * len(y)))
    index_tr = indexes[:index_split]
    index_te = indexes[index_split:]
    
    return x[index_tr], y[index_tr], x[index_te], y[index_te]


def plot_train_test(train_errors, test_errors, lambdas, degree):
    """
    train_errors, test_errors and lambas should be list (of the same size) the respective train error and test error for a given lambda,
    * lambda[0] = 1
    * train_errors[0] = RMSE of a ridge regression on the train set
    * test_errors[0] = RMSE of the parameter found by ridge regression applied on the test set
    
    degree is just used for the title of the plot.
    """
    plt.semilogx(lambdas, train_errors, color='b', marker='*', label="Train error")
    plt.semilogx(lambdas, test_errors, color='r', marker='*', label="Test error")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.title("Ridge regression for polynomial degree " + str(degree))
    leg = plt.legend(loc=1, shadow=True)
    leg.draw_frame(False)
    plt.savefig("ridge_regression")


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std






