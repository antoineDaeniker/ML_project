import numpy as np
import matplotlib.pyplot as plt
from .io_utils import *


def compute_loss(y, tx, w):
    """Calculate the loss.
        Can calculate the loss using mse or mae
    
    y: 1-D numpy array of labels
    tx: 2-D numpy array of data sample
    w: parameters of training procedure

    return:
        loss function
    """
    
    return 1/2 * np.mean((y - tx @ w)**2)  #MSE
    #return np.mean(abs(y - tx @ w)) #MAE


def compute_gradient(y, tx, w):
    """Compute the gradient
    
    y: 1-D numpy array of labels
    tx: 2-D numpy array of data sample
    w: parameters of training procedure

    return:
        gradient
    """
    return -tx.T @ (y - tx @ w) / len(y)


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


def build_poly(x, degree_start=-3, degree_end=8, include_half=True, include_cross_terms=True):
    """
        x: 2-D numpy array of data samples
        degree_start: the minimum polynomial degree in the feature equation
        degree_end: the maximum polynomial degree in the feature equation
        include_half: boolean specifies whether to include square rooted feature terms in polynomial
        include_cross_terms: boolean specifies whether to include bivariate feature terms in polynomial (e.g. X1X2, X1X3)

        returns: 2-D numpy array of data samples with modified features added
    """

    print(f'Building polynomial with {degree_start}, {degree_end}, {include_half}, {include_cross_terms}.')
    x = x.astype(float)
    poly = np.array(x, copy=True)
    zero_terms = 0
    for feat in x.T:
        for deg in range(degree_start, degree_end+1):
            # polynomial terms of degree one area already included
            # if feature contains 0 values do not raise to a negative exponent
            if deg != 0 and deg != 1:
                if deg > 0 or 0 not in feat:
                    poly = np.c_[poly, np.power(feat, deg)]
                else:
                    poly = np.c_[poly, np.zeros(len(feat))]
                    zero_terms += 1
        # if feature contains negative values, do not sqrt
        if include_half and not np.any(feat < 0):
            poly = np.c_[poly, np.power(feat, 0.5)]
        else:
            poly = np.c_[poly, np.zeros(len(feat))]
            # pad the final polynomial with 0 terms to keep the number of terms the same across training and testing
            zero_terms += 1
    if include_cross_terms:
        for feature_idx1 in range(len(x.T)):
            for feature_idx2 in range(feature_idx1+1, len(x.T)):
                poly = np.c_[poly, x.T[feature_idx1] * x.T[feature_idx2]]
    print(f'Generating polynomial with {poly.shape[1]} terms and {zero_terms} zero terms.')
    return poly


def build_k_indices(y, k_fold, seed=1):
    """build k indices for k-fold

    y: 1-D numpy array of labels
    k_fold: number of chunks
    seed: for randomize

    return:
        numpy array of k_indices
    """
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)
    

def split_cross_validation(y, x, k_indices, k, training_config=None):
    """ Cross validation function

    y: 1-D numpy array of labels
    tx: 2-D numpy array of data sample
    k_indices: contain chunks that contain sample indexes of data
    k: correspond to the chunck use for the validation set, other will be use for the train set
    training_config: trainging configuration

    return:
        x_tr: training set data
        y_tr: training set labels
        x_te: validation set data
        x_te: validation set labels
    """
    if training_config is None:
        raise ValueError('Please supply training config to cross validation function!')

    te_indices = k_indices[k]
    tr_indices = np.concatenate((k_indices[:k], k_indices[k+1:]), axis=0).reshape(-1)
    
    x_tr = x[tr_indices]
    y_tr = y[tr_indices]
    x_te = x[te_indices]
    y_te = y[te_indices]

    poly_config = dict(
        degree_start=training_config['start_degree'],
        degree_end=training_config['end_degree'],
        include_half=training_config['include_half'],
        include_cross_terms=training_config['include_cross_terms'],
    )

    x_tr = build_poly(x_tr, **poly_config)
    x_te = build_poly(x_te, **poly_config)

    return x_tr, y_tr, x_te, y_te


def sigmoid(t):
    """apply the sigmoid function on t
    
    t: prediction
    """
    return 1 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    """compute the loss: negative log likelihood
    
    y: 1-D numpy array of labels
    tx: 2-D numpy array of data sample
    w: parameters of training procedure

    return:
        negative log likelihood
    """
    pred = sigmoid(tx.dot(w))
    loss_i = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return -np.sum(loss_i)


def calculate_gradient(y, tx, w):
    """compute the gradient of loss
    
    y: 1-D numpy array of labels
    tx: 2-D numpy array of data sample
    w: parameters of training procedure

    return:
        gradient of likelihood function
    """
    pred = sigmoid(tx.dot(w))
    return tx.T.dot(pred - y)


#TODO VIEW THIS METHOD
    
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


def penalized_logistic_regression(y, tx, w, lambda_):
    """return the loss, gradient of likelihood function
    
    y: 1-D numpy array of labels
    tx: 2-D numpy array of data sample
    w: parameters of training procedure
    lambda_: regularize term

    return:
        loss: loss of function of likelihood function
        grad: gradient of function of likelihood function
    """
    loss = calculate_loss(y, tx, w) + lambda_ * np.linalg.norm(w)**2
    grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
    return loss, grad


def find_best_w(ws, losses):
    """ Find the best w from the list of all w during the train 
    
    ws: list (4, k_fold, #sample)list of w for each sub data and each chunk of the cross validation
    losses: losses for each w in the list

    return:
        w_best: best parameters for each sub data
    """
    w_best = []
    for loss, w in zip(losses, ws):
        idx = np.argmin(loss)
        w_best.append(w[idx])
    return w_best


def split_data_for_test_submit(ids, X_test, y, rmv_feat_list, training_config=None):
    """
    Split the data test in the same way we split the train data

    ids: indexes of sample of test set
    X_test: test set samples
    y: labels of test set
    rmv_feat_list: list of indexes of removed features of the traing set
    training_config: training configuration

    return:
        test_list: list of split data test
        y_list: list of test labels
        ids_list: list of test samples indexes
    """
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
        poly_config = dict(
            degree_start=training_config['start_degree'],
            degree_end=training_config['end_degree'],
            include_half=training_config['include_half'],
            include_cross_terms=training_config['include_cross_terms'],
        )
        sub_XData_norm,_ = normalize_data(sub_XData)
        sub_XData_poly = build_poly(sub_XData_norm, **poly_config)
        sub_y = y[bool_]

        ids_list.append(ids[bool_])
        test_list.append(sub_XData_poly)
        y_list.append(sub_y)

    return test_list, y_list, ids_list


def normalize_data(tX):
    """
    Normalize the data for each features

    tX: the data set

    return: 
        data_norm: normalized data
        std_0_feat_ind: potential indexes where the std of feature were 0
    """
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
        if std_features[i] != 0:
            f[f == -999] = mean_features[i]
            data_norm[:, i] = (f - mean_features[i]) / std_features[i]
        
    return np.array(data_norm), np.array(std_0_feat_ind)


def feature_correlation(tX, threshold, show_plot=False):
    """
    Compute features correlation pairs

    tX: data set
    threshold: if the correlation between two features is greater than treshold, we group them to keep only one of them
    show_plot: bool, True if we want to see correlations matrices

    return:
        data_reduce: data with grouped feature that were too corrolated
        corr_ind_to_throw: indexes of features we throw since we didn't need them (too corrolated with another one and we keep the other one)
        corr_ind_to_keep: indexes of features we kept and throw features they were corrolated
    """
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


def subdivide_data(tX, y):
    """
    Subdivide the data in different data set according to value in feature that took fiew different value 
                                                        (like PRI_jet_num which takes only 4 different value)

    tX: data set
    y: labels of data set

    return:
        data_list, y_list, feat_ind
    """
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


def delete_irr_features(data, threshold):
    """
    Delete features with high ratio of invalid values (-999)

    data: 2-D numpy array data set
    threshold: if ratio of invalid value per features is above threshold, delete this feature

    data_reduce: new data set without features with too much invalid values
    count_999_ind: list of features indexes with too much invalid values
    """
    #Remove irrelevant feature, for each data set
    count_999 = []
    for feat in data.T:
        count_999.append(len(feat[feat == -999]) / len(feat))
    
    count_999_ind = np.squeeze(np.argwhere(np.array(count_999) > threshold))
    data_reduce = np.delete(data, np.array(count_999_ind), axis=1)
    
    return data_reduce, count_999_ind


def data_train_preprocessing(tX, y, threshold_irr, threshold_corr, show_plot=False):
    """
    Compute all process method on data set

    tX: 2-D numpy array data set
    y: 1-D numpy array labels
    threshold_irr: threshold for irrelevant features
    threshold_corr: threshold for corrolated features
    show_plot: bool, True if we want to see correlations matrices

    return:
        data_reduce_list: list of processed sub data set
        y_list: list of labels
        rmv_feat_idx_list: list of removed indexes for each sub data
    """
    
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
    
    
def get_accuracy(y_pred, y_gt):
    """
    Get the accuracy between predictions and ground truths labels

    y_pred: predictions labels
    y_gt: ground truth labels

    return:
        accuracy
    """
    accuracy = len(y_pred[y_pred == y_gt]) * 100 / y_gt.shape[0]
    return accuracy

