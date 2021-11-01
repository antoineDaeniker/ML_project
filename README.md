
# ML_project


## 1. Librairies

We use `matplotlib`, `numpy` as librairies so make sure you have them installed. To make the predictions we only require `numpy`.

## 2. Data 

The current data path points to the `data` folder which contains `train.csv` and `test.csv`. Make sure the right data path is set in the loading function `load_csv_data` in `$utils\io.py`. 

## 3. Configuartion setting

You can change configurations you want in `model.py` file specifically in the `get_run_configs` function for cross validation. Specifically for generating predictions use the `training_config` variable in the main function.

*NOTE*: In order to obtain our final results you can use our final configuration.

## 4. Running the project

To get the same predictions, just run the `run.py` file. The submission csv will be generated with the timestamp of the run. 

## 5. Results

To get the results, a file will be automatically created in  `$Data/submissions` to make your submission in AIcrowd.

*NOTE*: If relevant, change the name of your results file in `create_csv_submission` function implemented in `$utils\io.py`. 



## 6. Code architecture:

#### Individual scripts:

*  `implementations.py` : contains the main methods used
*  `model.py`: - creates and run different possible configurations (different methods, different parameters..)
            - generates model on train data based on the config input
            - tests model on test data calculating accuracy
* `run.py`: runs the model

#### Folders: 

1. Folder `data file`: contains training and test data

2. Folder `utils file`: general file for functions used in main functions
* `io_utils.py`: loads data, makes predictions based on the test data using the model, and outputs results in a file for submission to Kaggle or AIcrowd
* `preprossessing_utils.py` : functions used for preprocessing 
* `implementation_utils.py` : functions used for implementations and running the model 

3. Folder `figs` : Contains figures for visualisations 

##### Notebooks: 
for testing
