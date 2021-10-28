import numpy as np
import logging

from utils.implementation_utils import delete_irr_features, feature_correlation, normalize_data

logger = logging.getLogger(__name__)

def update_labels(y):
    """ Change y labels from y E {-1, 1} to y E {0, 1} """
    y[np.where(y == -1)] = 0
    return y


def preprocess_train_data(X, y):
    logger.info('Preprocessing data')

    data_irr, irr_ind = delete_irr_features(X, 0.5)
    data_irr_corr, corr_ind,_ = feature_correlation(data_irr, 0.9)
    data_irr_corr_norm, norm_ind = normalize_data(data_irr_corr)

    rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
    rmv_idx = np.insert(rmv_idx, -1, norm_ind)
    rmv_idx = np.unique(rmv_idx)

    y = update_labels(y)

    return data_irr_corr_norm, y, rmv_idx

