import numpy as np
import matplotlib.pyplot as plt
from .io_utils import *

""" TODO split into multiple util files"""

def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient"""
    loss = calculate_loss(y, tx, w) + lambda_ * np.linalg.norm(w)**2
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, grad

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


def calculate_least_squares(y, tx):
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
    
def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
    
def cross_validation_for_rgr(y, tx, k_fold, k, lambda_, initial_w, max_iters, gamma, seed=1):
    """return the loss of ridge regression."""
    rmse_tr = []
    rmse_te = []
    k_indices = build_k_indices(y, k_fold, seed)
    # ***************************************************
    # get k'th subgroup in test, others in train
    # ***************************************************
    for k in range(k_fold):

        te_indice = k_indices[k]
        tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
        tr_indice = tr_indice.reshape(-1)
        y_te = y[te_indice]
        y_tr = y[tr_indice]
        tx_te = tx[te_indice]
        tx_tr = tx[tr_indice]
    
        w, loss = reg_logistic_regression(y_tr, tx_tr, lambda_ , initial_w, max_iters, gamma)
        
        loss_tr = np.sqrt(2 * calculate_loss(y_tr, x_tr, w))
        loss_te = np.sqrt(2 * calculate_loss(y_te, x_te, w))

        rmse_tr.append(loss_tr)
        rmse_te.append(loss_te)
    return rmse_tr, rmse_te, w
    
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


def split_cross_validation(y, x, k_indices, k):
    """return the loss of ridge regression."""
    
    te_indices = k_indices[k]
    tr_indices = np.concatenate((k_indices[:k], k_indices[k+1:]), axis=0).reshape(-1)
    
    x_tr = x[tr_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    y_te = y[te_indices]

    return x_tr, y_tr, x_te, y_te


def cross_validation(y, x, k_indices, k, lambda_, degree):
    """return the loss of ridge regression."""
    
    te_indices = k_indices[k]
    tr_indices = np.concatenate((k_indices[:k], k_indices[k+1:]), axis=0).reshape(-1)
    
    x_tr = x[tr_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    y_te = y[te_indices]

    tx_tr = build_poly(x_tr, degree)
    tx_te = build_poly(x_te, degree)

    w = ridge_regression(y_tr, tx_tr, lambda_)
    
    loss_tr = np.sqrt(2 * compute_loss(y_tr, tx_tr, w))
    loss_te = np.sqrt(2 * compute_loss(y_te, tx_te, w))
    return loss_tr, loss_te


def sigmoid(t):
    """apply the sigmoid function on t."""
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood."""
    pred = sigmoid(tx.dot(w))
    loss_i = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return -np.sum(loss_i)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y)


def learning_by_gradient_descent(y, tx, w, gamma):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = calculate_loss(y, tx, w)
    grad = calculate_gradient(y, tx, w)
    w = w - gamma * grad
    return loss, w


def logistic_regression_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    threshold = 1e-8
    gamma = 0.01
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_gradient_descent(y, tx, w, gamma)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_gradient_descent", True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))

    
def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    sig = sigmoid(tx.dot(w))
    pred = np.diag(sig.T[0])
    return tx.T.dot(np.multiply(pred, (1 - pred))).dot(tx)


def calculate_loss_grad_hess(y, tx, w):
    """return the loss, gradient, and Hessian."""
    return calculate_loss(y, tx, w), calculate_gradient(y, tx, w), calculate_hessian(y, tx, w)


def learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss, grad, hess = calculate_loss_grad_hess(y, tx, w)
    w = np.linalg.solve(hess, hess.dot(w) - gamma * grad)
    return loss, w


def logistic_regression_newton_method_demo(y, x):
    # init parameters
    max_iter = 100
    threshold = 1e-8
    lambda_ = 0.1
    gamma = 1.
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_newton_method(y, tx, w, gamma)
        # log info
        if iter % 1 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_newton_method",True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))



def learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent, using the penalized logistic regression.
    Return the loss and updated w.
    """
    loss, grad = penalized_logistic_regression(y, tx, w, lambda_)
    w = w - gamma * grad
    return loss, w


def logistic_regression_penalized_gradient_descent_demo(y, x):
    # init parameters
    max_iter = 10000
    gamma = 0.01
    lambda_ = 0.1
    threshold = 1e-8
    losses = []

    # build tx
    tx = np.c_[np.ones((y.shape[0], 1)), x]
    w = np.zeros((tx.shape[1], 1))

    # start the logistic regression
    for iter in range(max_iter):
        # get loss and update w.
        loss, w = learning_by_penalized_gradient(y, tx, w, gamma, lambda_)
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    # visualization
    visualization(y, x, mean_x, std_x, w, "classification_by_logistic_regression_penalized_gradient_descent",True)
    print("loss={l}".format(l=calculate_loss(y, tx, w)))




def find_best_w(ws, losses):
    """ Find the best w from the list of all w during the train """
    w_best = []
    for loss, w in zip(losses, ws):
        idx = np.argmin(loss)
        w_best.append(w[idx])
    return w_best


def split_data_for_test_submit(ids, X_test, y, rmv_feat_list):
    #Find the nb of different value in per feature
    nb_diff_values = []
    for feat in X_test.T:
        nb_diff_values.append(len(np.unique(feat)))
        
    #Create the different dataset w.r.t those different value (and remove the feature concern)
    feat_ind = np.argmin(nb_diff_values)
    feat_values = np.unique(X_test[:, feat_ind])
    ids_list = []
    test_list = []
    y_list = []
    for i, (val, rmv_feat_indx) in enumerate(zip(feat_values, rmv_feat_list)):
        bool_ = X_test[:, feat_ind] == val

        sub_XData = X_test[bool_]
        sub_XData = np.delete(sub_XData, rmv_feat_indx, axis=1)
        sub_XData_poly = build_poly(sub_XData, 8)
        sub_XData,_ = normalize_data(sub_XData_poly)
        sub_y = y[bool_]

        ids_list.append(ids[bool_])
        test_list.append(sub_XData)
        y_list.append(sub_y)

    return test_list, y_list, ids_list


def data_for_test_submit(X_test, rmv_feat):
    
    XData = np.delete(X_test, rmv_feat, axis=1)
    XData_poly = build_poly(XData, 8)
    XData_norm,_ = normalize_data(XData_poly)
    new_X_test = np.concatenate((np.ones(XData_norm.shape[0])[:, np.newaxis], XData_norm), axis=1)

    return new_X_test



############################################
########### NORMALIZE THE DATA #############
############################################
def normalize_data(tX):
    temp = tX.copy()
    temp[temp == -999] = 0
    mean_features = np.mean(temp, axis=0)
    std_features = np.std(temp, axis=0)
    std_0_feat_ind = np.squeeze(np.argwhere(std_features <= 0.001))
    #mean_features = np.delete(mean_features, std_0_feat_ind)
    #std_features = np.delete(std_features, std_0_feat_ind)
    data_reduce = np.delete(tX, std_0_feat_ind, axis=1)
    data_norm = np.zeros(tX.shape)
    for i, f in enumerate(tX.T):
        if std_features[i] != 0 :
            f[f == -999] = mean_features[i]
            data_norm[:, i] = (f - mean_features[i]) / std_features[i]
        
    return np.array(data_norm), np.array(std_0_feat_ind)


############################################
###### FEATURES CORRELATION SELECTION ######
############################################
def feature_correlation(tX, threshold, show_plot=False):
    corr_mat = np.ma.corrcoef(tX, rowvar=False)
    #corr_mat[corr_mat == np.nan] = 0
    
    if show_plot:
        plt.imshow(np.abs(corr_mat), cmap='Blues')
        plt.colorbar()
        plt.show()
        
    corr_indices = np.argwhere(np.abs(np.triu(corr_mat - np.eye(tX.shape[1]))) > threshold)
    
    unique_ind1 = np.unique(corr_indices[:, 0])
    unique_ind2 = np.unique(corr_indices[:, 1])
    len1 = len(unique_ind1)
    len2 = len(unique_ind2)
    corr_ind_reduce_short = (unique_ind1, unique_ind2)[len(unique_ind1) > len(unique_ind2)]
    corr_ind_reduce_big = (unique_ind1, unique_ind2)[len(unique_ind1) <= len(unique_ind2)]
    corr_ind_to_keep = []
    for ind in corr_ind_reduce_short:
        is_in = np.isin(ind, corr_ind_reduce_big)
        if not is_in:
            corr_ind_to_keep.append(ind)
            
    all_ind = np.unique(corr_indices.flatten())
    corr_ind_to_throw = np.setdiff1d(all_ind, corr_ind_to_keep)
    
    data_reduce = np.delete(tX, corr_ind_to_throw, axis=1)
    
    return data_reduce, corr_ind_to_throw, np.array(corr_ind_to_keep)


###############################################################
###### DIVIDE DATA W.R.T FEATURE WITH LOW VARIATION DATA ######
###############################################################
def subdivide_data(tX, y):
    #Find the nb of different value in per feature
    nb_diff_values = []
    for feat in tX.T:
        nb_diff_values.append(len(np.unique(feat)))
        
    #Create the different dataset w.r.t those different value (and remove the feature concern)
    feat_ind = np.argmin(nb_diff_values)
    feat_values = np.unique(tX[:, feat_ind])
    data_list = []
    y_list = []
    for i, val in enumerate(feat_values):
        data_list.append(tX[tX[:, feat_ind] == val])
        y_list.append(y[tX[:, feat_ind] == val])
    
    return data_list, y_list, feat_ind


################################################################
### DELETE FEATURE WITH MORE THAN THRESHOLD OF CORRUPT DATA ####
################################################################
def delete_irr_features(data, threshold):
    #Remove irrelevant feature, for each data set
    count_999 = []
    for feat in data.T:
        count_999.append(len(feat[feat == -999]) / len(feat))
    
    count_999_ind = np.squeeze(np.argwhere(np.array(count_999) > threshold))
    data_reduce = np.delete(data, np.array(count_999_ind), axis=1)
    
    return data_reduce, count_999_ind

################################################################
#################### PRE-PROCESS THE DATA ######################
################################################################
def data_train_preprocessing(tX, y, threshold_irr, threshold_corr, show_plot=False):
    
    data_list, y_list, feat_ind = subdivide_data(tX, y)
    
    data_reduce_list = []
    rmv_feat_idx_list = []
    for i, data in enumerate(data_list):
        data_reduce, irr_idx = delete_irr_features(data, threshold_irr)
        data_reduce, corr_idx, _ = feature_correlation(data_reduce, threshold_corr)
        data_reduce, std_0_feat_ind = normalize_data(data_reduce)

        print("train data {} shape : {}".format(i, data_reduce.shape))
        data_reduce_list.append(data_reduce)

        rmv_feat_idx = np.unique(np.concatenate((irr_idx, corr_idx)))
        rmv_feat_idx = np.insert(rmv_feat_idx, -1, std_0_feat_ind)
        rmv_feat_idx = np.insert(rmv_feat_idx, -1, feat_ind)
        rmv_feat_idx_list.append(np.unique(rmv_feat_idx))
        
    return data_reduce_list, y_list, rmv_feat_idx_list


def data_test_preprocessing(tX_test, y_test, rmv_feat_idx_list, threshold_irr, threshold_corr, show_plot=False):
    
    data_list_test, y_list_test, feat_ind = subdivide_data(tX_test, y_test)
    
    new_data_test_list = []
    for rmv_feat_idx, data_test in zip(rmv_feat_idx_list, data_list_test):
        data_test_reduce = np.delete(data_test, rmv_feat_idx, axis=1)
        norm_data_test_reduce, _ = normalize_data(data_test_reduce)
        
        print(norm_data_test_reduce.shape)
        new_data_test_list.append(norm_data_test_reduce)
        
    return new_data_test_list, y_list_test

    
    
def get_accuracy(y_pred, y_gt):
    accuracy = len(y_pred[y_pred == y_gt]) * 100 / y_gt.shape[0]
    return accuracy


"""
    # Save weights
    np.savetxt('sgd_model.csv', np.asarray(sgd_ws), delimiter=',')

    # Load weights
    sgd_ws = np.loadtxt('sgd_model.csv', delimiter=',')

"""

