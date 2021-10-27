#%%
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
import datetime

from proj1_helpers import *

DATA_TRAIN_PATH = '/Users/AntoineDaeniker/Documents/EPFL/Master 1/ML_course/projects/project1/data/train.csv' # TODO: download train data and supply path here 
y, X, Xt, ids = load_csv_data(DATA_TRAIN_PATH)
print(y.shape, X.shape)

DATA_TEST_PATH = '/Users/AntoineDaeniker/Documents/EPFL/Master 1/ML_course/projects/project1/data/test.csv' # TODO: download train data and supply path here 
y_test, X_test, Xt_test, ids_test = load_csv_data(DATA_TEST_PATH)


#%%
##################### DATA TRAIN PROCESSING #####################
data_irr, irr_ind = delete_irr_features(X, 0.5)
data_irr_corr, corr_ind,_ = feature_correlation(data_irr, 0.9)
data_irr_corr_norm, norm_ind = normalize_data(data_irr_corr)

rmv_idx = np.unique(np.concatenate((irr_ind, corr_ind)))
rmv_idx = np.insert(rmv_idx, -1, norm_ind)
rmv_idx = np.unique(rmv_idx)
print("DATA TRAIN PROCESSING DONE")
###########################################################

##################### TRAINING ############################
# Define the parameters of the algorithm.
max_iters = 800
gamma = 0.1
batch_size = 1

# Initialization
w_initial = np.zeros(data_irr_corr_norm.shape[1])

# Start SGD.
start_time = datetime.datetime.now()
sgd_losses, sgd_ws = gradient_descent(
    y, data_irr_corr_norm, w_initial, max_iters, gamma)

for i in rmv_idx:
    sgd_ws = np.insert(sgd_ws, i, 0)
print("\nRESULTING W : ", sgd_ws)

end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("SGD: execution time={t:.3f} seconds".format(t=exection_time))
print("TRAINING DONE")
###########################################################

################# DATA TEST PROCESSING ####################
#data_test = np.delete(X_test, rmv_idx, axis=1)
print("DATA TEST PROCESSING DONE")
###########################################################

######################## ACCURACY #########################
acc = accuracy(sgd_ws, X_test, y_test, y_test.shape[0])
print("ACCURACY : ", acc)
# %%
