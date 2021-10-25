import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

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


############################################
########### NORMALIZE THE DATA #############
############################################
def normalize_data(tX):
    temp = tX.copy()
    temp[temp == -999] = 0
    mean_features = np.mean(temp, axis=0)
    std_features = np.std(temp, axis=0)
    std_0_feat_ind = np.squeeze(np.argwhere(std_features <= 0.001))
    mean_features = np.delete(mean_features, std_0_feat_ind)
    std_features = np.delete(std_features, std_0_feat_ind)
    data_reduce = np.delete(tX, std_0_feat_ind, axis=1)
    data_norm = np.zeros(data_reduce.shape)
    for i, f in enumerate(data_reduce.T):
        f[f == -999] = mean_features[i]
        data_norm[:, i] = (f - mean_features[i]) / std_features[i]
        
    return np.array(data_norm), np.array(std_0_feat_ind)


############################################
###### FEATURES CORRELATION SELECTION ######
############################################
def feature_correlation(tX, threshold, show_plot=False):
    corr_mat = np.corrcoef(tX, rowvar=False)
    
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
        data_list.append(np.delete(tX[tX[:, feat_ind] == val], feat_ind, axis=1))
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
    

def accuracy(w, data, y, y_test_len):
    pred = predict_labels(w, data)
    accuracy = len(pred[pred == y]) * 100 / y_test_len
        
    return accuracy
    
    


