import numpy as np
import logging

from utils.implementation_utils import delete_irr_features, feature_correlation, normalize_data, subdivide_data

logger = logging.getLogger(__name__)

def update_labels(y):
    """ Change y labels from y E {-1, 1} to y E {0, 1} """
    y[np.where(y == -1)] = 0
    return y


def preprocess_train_data(X, y):
    logger.info('Preprocessing data')

    _, irr_ind = delete_irr_features(X, 0.5)
    _, corr_ind,_ = feature_correlation(X, 0.9)
    _, norm_ind = normalize_data(X)

    rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
    rmv_idx = np.insert(rmv_idx, -1, norm_ind)
    rmv_idx = np.unique(rmv_idx)
    print(rmv_idx)
    data_reduce = np.delete(X, rmv_idx, axis=1)
    data_irr_corr_norm,_ = normalize_data(data_reduce)

    y = update_labels(y)

    return data_irr_corr_norm, y, rmv_idx


def preprocess_train_data_split(X, y):
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
        print(rmv_idx)
        data_reduce = np.delete(X, rmv_idx, axis=1)
        data_irr_corr_norm,_ = normalize_data(data_reduce)

        y = update_labels(y)

        data_split.append(data_irr_corr_norm)
        y_split.append(y)
        rmv_idx_split.append(rmv_idx)

    return data_split, y_split, rmv_idx_split

