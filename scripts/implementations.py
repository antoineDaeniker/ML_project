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
        print("Gradient Descent({bi}/{ti}): loss={l}".format(
              bi=n_iter, ti=max_iters - 1, l=loss), end='\r')

    return losses, ws[len(ws) - 1]


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
    poly = np.ones((x.shape[0], 1))
    for feat in x.T:
        for deg in range(1, degree+1):
            poly = np.c_[poly, np.power(feat, deg)]
    return poly


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


def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

############################################
########### DATA PRE-PROCESSING ############
############################################
def data_process(tX):
    temp = tX.copy()
    tX_norm = np.zeros(tX.shape)
    temp[temp == -999] = 0
    mean_features = np.mean(temp, axis=0)
    std_features = np.std(temp, axis=0)
    for i, f in enumerate(tX.T):
        f[f == -999] = mean_features[i]
        #tX_norm[:, i] = f
        tX_norm[:, i] = (f - mean_features[i]) / std_features[i]
        
    return np.array(tX_norm)






