import numpy as np
import matplotlib.pyplot as plt
from implementations import *
import datetime

from proj1_helpers import *
DATA_TRAIN_PATH = '/Users/AntoineDaeniker/Documents/EPFL/Master 1/ML_course/projects/project1/data/train.csv' # TODO: download train data and supply path here 
y, tX, tXt, ids = load_csv_data(DATA_TRAIN_PATH)
print(y.shape, tX.shape)


###### PRE_PROCESSING #######
temp = tX.copy()
tX_norm = np.zeros(tX.shape)
temp[temp == -999] = 0
mean_features = np.mean(temp, axis=0)
std_features = np.std(temp, axis=0)
for i, f in enumerate(tX.T):
    f[f == -999] = mean_features[i]
    #tX_norm[:, i] = f
    tX_norm[:, i] = (f - mean_features[i]) / std_features[i]

tXt_norm = np.c_[np.ones(len(y)) / len(y), tX_norm]


##### TRAINING #####
max_iters = 200
gamma = 2*10**(-3)
batch_size = 1

# Initialization
w_initial = np.random.rand(tXt_norm.shape[1])

# Start SGD.
start_time = datetime.datetime.now()
sgd_losses, sgd_ws = gradient_descent(
    y, tXt_norm, w_initial, max_iters, gamma)
end_time = datetime.datetime.now()

# Print result
exection_time = (end_time - start_time).total_seconds()
print("\nSGD: execution time={t:.3f} seconds, gamma = {gamma}\n".format(t=exection_time, gamma=gamma))


##### PLOT LOSSES #####
plt.plot(np.arange(max_iters), sgd_losses)



DATA_TEST_PATH = '/Users/AntoineDaeniker/Documents/EPFL/Master 1/ML_course/projects/project1/data/test.csv' # TODO: download train data and supply path here 
y_test, tX_test, tXt_test, ids_test = load_csv_data(DATA_TEST_PATH)


###### PRE_PROCESSING #######
temp_test = tX_test.copy()
tX_norm_test = np.zeros(tX_test.shape)
temp_test[temp_test == -999] = 0
mean_features_test = np.mean(temp_test, axis=0)
std_features_test = np.std(temp_test, axis=0)
for i, f in enumerate(tX_test.T):
    f[f == -999] = mean_features_test[i]
    tX_norm_test[:, i] = (f - mean_features_test[i]) / std_features_test[i]
    
tXt_norm_test = np.c_[np.ones(len(y_test)) / len(y_test), tX_norm_test]


pred_train = predict_labels(sgd_ws, tXt_norm)
print(len(pred_train[abs(pred_train-y) == 0]) * 100 / len(y))

pred_test = predict_labels(sgd_ws, tXt_norm_test)
print(len(pred_test[abs(pred_test-y_test) == 0]) * 100 / len(y_test))