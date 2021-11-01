import numpy as np
import logging

from utils.implementation_utils import build_poly, delete_irr_features, feature_correlation, normalize_data, split_data_for_test_submit, subdivide_data
from utils.io_utils import predict_labels

logger = logging.getLogger(__name__)

def update_labels(y):
    """ 
    y: 1-D numpy array, change y labels from y E {-1, 1} to y E {0, 1} 
    """

    y[np.where(y == -1)] = 0
    return y


def preprocess_train_data(X, y, update_labels = True):
    """
    Get feature indexes of irrelevant features (those with to much invalid values), group corrolate features and 
    get features indexes of features with 0 standard deviation

    X: 2-D numpy array with the data sample
    y: 1-D numpy array with labels of data set
    update_labels: bool, True if we change y labels from {-1, 1} to {0, 1}

    return: 
        data_irr_corr_norm: the new process data set
        y: the updated labels if update_labels was set to true
        rmv_idx: array with indexes of feature to remove
    """

    logger.info('Preprocessing data')

    _, irr_ind = delete_irr_features(X, 0.5)
    _, corr_ind,_ = feature_correlation(X, 0.9)
    _, norm_ind = normalize_data(X)

    rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
    rmv_idx = np.insert(rmv_idx, -1, norm_ind)
    rmv_idx = np.unique(rmv_idx)
    print(rmv_idx)
    data_reduce = np.delete(X, rmv_idx, axis=1)
    data_reduce = build_poly(data_reduce)
    data_irr_corr_norm,_ = normalize_data(data_reduce)
    data_irr_corr_norm = np.concatenate((np.ones(data_irr_corr_norm.shape[0])[:, np.newaxis], data_irr_corr_norm), axis=1)

    if update_labels:
        y = update_labels(y)

    return data_irr_corr_norm, y, rmv_idx


def preprocess_train_data_split(X, y, update_label=False):
    """
    Similar method as preprocess_train_data but do it for all sub data set

    X: 2-D numpy array with the data sample
    y: 1-D numpy array with labels of data set
    update_labels: bool, True if we change y labels from {-1, 1} to {0, 1}

    return: 
        data_split: list of new process data set
        y_split: list of updated labels if update_labels was set to true
        rmv_idx_split: list of array with indexes of feature to remove
    """

    logger.info('Preprocessing data')
    X_list, y_list, feat_ind = subdivide_data(X, y)
    data_split = []
    y_split = []
    rmv_idx_split = []

    for X, y in zip(X_list, y_list):
        _, irr_ind = delete_irr_features(X, 0.5)
        _, corr_ind,_ = feature_correlation(X, 0.9)
        _, norm_ind = normalize_data(X)

        rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
        rmv_idx = np.insert(rmv_idx, -1, norm_ind)
        rmv_idx = np.insert(rmv_idx, -1, feat_ind)
        rmv_idx = np.unique(rmv_idx)
        print('Removed features indexes : ', rmv_idx)
        data_reduce = np.delete(X, rmv_idx, axis=1)
        data_irr_corr_norm,_ = normalize_data(data_reduce)

        if update_label:
            y = update_labels(y)

        data_split.append(data_irr_corr_norm)
        y_split.append(y)
        rmv_idx_split.append(rmv_idx)

    return data_split, y_split, rmv_idx_split


def make_prediction_split_for_submission(y_te, X_te, ids_te, rmv_idx_list, ws_best):
    """
    Reconstruc the y labels in the order of the indexes sample of the test set

    y_te: 1-D numpy array of labels of the test set
    X_te: 2-D numpy array of sample data of the test set
    ids_te: 1-D numpy array of indexes of test set
    rmv_idx_list: list of removed indexes features for each sub data
    ws_best: w parameter results of the training procedure

    return:
        new_y_pred: 1-D numpy array of predictions
    """
    y_pred_list = []
    X_test_list, y_test_list, ids_list = split_data_for_test_submit(ids_te, X_te, y_te, rmv_idx_list)
    for ws, rmv_idx, sub_X_test in zip(ws_best, rmv_idx_list, X_test_list):
        sub_X_test = np.concatenate((np.ones(sub_X_test.shape[0])[:, np.newaxis], sub_X_test), axis=1)
        y_pred = predict_labels(ws, sub_X_test)
        y_pred_list.append(y_pred)
    y_pred_conc = np.concatenate((y_pred_list[0], y_pred_list[1], y_pred_list[2] ,y_pred_list[3]), axis=0)[:, np.newaxis]
    ids_conc = np.concatenate((ids_list[0], ids_list[1], ids_list[2] ,ids_list[3]), axis=0)[:, np.newaxis]
    ids_y_conc = np.concatenate((ids_conc, y_pred_conc), axis=1)
    sorted_y_conc = ids_y_conc[ids_y_conc[:, 0].argsort()]
    new_y_pred = np.squeeze(np.delete(sorted_y_conc, 0, axis=1))

    return new_y_pred

