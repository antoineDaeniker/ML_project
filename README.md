# ML_project

Overview of the code's architecture:

#Individual scripts:

* implementations.py : contains the main methods used
* model.py: - creates and run different possible configurations (different methods, different parameters..)
            - generate model on train data based on the config input
            - tests model on test data calculating accuracy
* run.py: runs the model

#Folders: 

* data file: contains training and test data
* utils file: general file for functions used in main functions
            - io_utils.py: loads data, makes predictions based on the test data using the model, and outputs results in a file for submission to Kaggle or AIcrowd
            - preprossessing_utils.py : functions used for preprocessing 
            - implementation_utils.py : functions used for implementations and running the model 

* figs file: Contains figures for visualisations 

#notebooks: for testing

            
