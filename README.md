# ML_project

todo:
* double check the main methods (Umer) 
* cross validation (hyper parameter search) - correlation threshold can be a hyper parameter 
* mean change (Antoine) drop the 
* drop the irrelevant features (Antoine) - 0, 4, 5, 6, 12, 22-27 because invalid values  3, 9, 21  because high correlation (14 drop)
* Adam optimizer

learning rates: 1, 2, 3
num_to_drop: 6, 7, 8



Structure & Files description:

#Individual scripts:

* implementations.py : contains the main methods used
* model.py: - creates and run different possible configurations (different methods, different parameters..)
            - generate model on train data based on the config input
            - tests model on test data calculating accuracy
* run.py: runs the model

#Folders's scripts

* utils file: general file for functions used in main functions
            - io_utils.py: loads data, makes predictions based on the test data using the model, and outputs results in a file for submission to Kaggle or AIcrowd
            - preprossessing_utils.py : functions used for preprocessing 
            - implementation_utils.py : functions used for implementations and running the model #TODO Split??
            
