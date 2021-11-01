
# ML_project


## 1. Librairies

We use `matplotlib`, `numpy` as librairies so make sure you have them installed.

## 2. Data 

Make sure the `data` folder is available in the project and that the right data path is set in the loading function `load_csv_data` in `$utils\io.py`. 

## 3. Configuartion setting

You can change configurations you want in `model.py` file specifically in the `get_run_configs` function.

*NOTE*: In order to abtain our results you can use our final configuration.

## 4. Running the project

To run the project, just run the `run.py` file.

## 5. Results

To get the results, a file will be automatically created in  `$Data/submissions` to make your submission in AIcrowd.

*NOTE*: Change the name of your results file in `create_csv_submission` function implmented in `$utils\io.py`. 



## 6. Code's architecture:

# Individual scripts:

* `implementations.py` : contains the main methods used
* `model.py`: - creates and run different possible configurations (different methods, different parameters..)
            - generate model on train data based on the config input
            - tests model on test data calculating accuracy
* `run.py`: runs the model

# Folders: 

1. Folder `data file`: contains training and test data

2. Folder `utils file`: general file for functions used in main functions
            * io_utils.py: loads data, makes predictions based on the test data using the model, and outputs results in a file for submission to Kaggle or AIcrowd
            * preprossessing_utils.py : functions used for preprocessing 
            *implementation_utils.py : functions used for implementations and running the model 

3. Folder `figs` : Contains figures for visualisations 

# Notebooks: 
for testing
