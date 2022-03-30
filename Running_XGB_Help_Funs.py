import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from xgboost import XGBRegressor


def main_XGB_all_directions(train, test):
    
    '''
    This function
    '''
    
    ## Defining all the directions
    directions = train['direction'].unique()
    
    ## Defining list to store results
    results_all_directions = list()
    
    for i in range(0, len(directions)):
        
        ## Subsetting train based on directions
        temp_train = train[train['direction'] == directions[i]].reset_index(drop = True)
        
        ## Appending results 
        print(main_XGB_all_directions_help(temp_train))
#         results_all_directions.append(main_XGB_all_directions_help(temp_train))
        
        
def main_XGB_all_directions_help(train):
        
    ## Defining locations 
    x_values = train['x'].unique()
    y_values = train['y'].unique()
    
    ## Defining list to store results
    results_all_locations = list()
    
    for i in range(0, len(x_values)):
        
        for j in range(0, len(y_values)):
            
            temp_train = train[(train['x'] == x_values[i]) & (train['y'] == y_values[j])].reset_index(drop = True)
            print(temp_train.shape)
            
            

def main_XGB_all_directions_help_help(train):            
    
    ## Defining train & validation trainsets
    X_train = train.loc[0:13023, ['day', 'hour', 'minute']]
    Y_train = train.loc[0:13023, ['congestion']]

    X_val = train.loc[13023:13059, ['day', 'hour', 'minute']]
    Y_val = train.loc[13023:13059, ['congestion']]
    
    ## Defining the hyper-parameter grid
    XGBoost_param_grid = {'n_estimators': [300],
                          'max_depth': [5, 7],
                          'min_child_weight': [5, 7, 10],
                          'learning_rate': [0.01],
                          'gamma': [0.3, 0.1],
                          'subsample': [0.8, 1],
                          'colsample_bytree': [1]}

    ## Performing grid search with 5 folds
    XGBoost_grid_search = GridSearchCV(XGBRegressor(), XGBoost_param_grid, cv = 5, scoring = 'neg_mean_squared_error').fit(X_train, Y_train)

    ## Extracting the best model
    XGBoost_md = XGBoost_grid_search.best_estimator_

    ## Predicting on validation & test 
    XGBoost_pred = XGBoost_md.predict(X_val)
    
            
            
            