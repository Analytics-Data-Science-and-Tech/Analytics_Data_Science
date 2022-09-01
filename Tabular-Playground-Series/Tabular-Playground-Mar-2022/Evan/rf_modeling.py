import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor


def main_rf(train, test):
    
    ## Defining a list of all directions
    directions = train['direction'].unique()
    
    ## Defining lists to store results
    results_all_directions_val = list()
    results_all_directions_test = list()
    
    for i in range(0, len(directions)):
        
        ## Printing message to confirm direction
        print('Working on direction:', directions[i])
        
        ## Subsetting the data based on direction
        train_temp = train[train['direction'] == directions[i]].reset_index(drop = True)
        test_temp = test[test['direction'] == directions[i]].reset_index(drop = True)
        
        ## Appending results
        results = main_rf_help(train_temp, test_temp)
        results_all_directions_val.append(results[0])
        results_all_directions_test.append(results[1])
            
    return [pd.concat(results_all_directions_val), pd.concat(results_all_directions_test)]  



def main_rf_help(train, test):
    
    ## Defining a list of all locations
    x_locations = train['x'].unique()
    y_locations = train['y'].unique()
    
    ## Defining lists to store results
    results_all_locations_val = list()
    results_all_locations_test = list()
    
    for i in range(0, len(x_locations)):
        
        for j in range(0, len(y_locations)):
            
            ## Printing message to confirm location
            print('Working on location: (', x_locations[i], ',', y_locations[j], ')')
            
            ## Subsetting the data based on location
            train_temp = train[(train['x'] == x_locations[i]) & (train['y'] == y_locations[j])].reset_index(drop = True)
            test_temp = test[(test['x'] == x_locations[i]) & (test['y'] == y_locations[j])].reset_index(drop = True)
            
            ## Ignoring subsets with no observations
            if (train_temp.shape[0] == 0):
                
                continue
            
            ## Modeling building and prediction at location (x, y)
            results = main_rf_help_help(train_temp, test_temp)
            results_all_locations_val.append(results[0])
            results_all_locations_test.append(results[1])
            
    return [pd.concat(results_all_locations_val), pd.concat(results_all_locations_test)]



def main_rf_help_help(train, test):            
    
    ## Defining train, validation, and test datasets
    X_train = train.loc[0:13023, ['day', 'hour', 'minute']]
    Y_train = train.loc[0:13023, ['congestion']]

    X_val = train.loc[13023:13059, ['day', 'hour', 'minute']]
    Y_val = train.loc[13023:13059, ['congestion']]
    
    X_test = test[['day', 'hour', 'minute']]
    
    ## Defining the hyper-parameter grid
    rf_param_grid = {'n_estimators': [100, 300, 500], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 6, 10], 'min_samples_leaf': [1, 5, 9], 'n_jobs': [-1]}

    ## Performing grid search with 5 folds
    rf_grid_search = GridSearchCV(RandomForestRegressor(), rf_param_grid, cv = 5, scoring = 'neg_mean_absolute_error').fit(X_train, Y_train)

    ## Extracting the best model
    rf_md = rf_grid_search.best_estimator_

    ## Predicting on validation & test 
    rf_val_pred = rf_md.predict(X_val)
    rf_test_pred = rf_md.predict(X_test)
    
    ## Appending predictions on validation and test
    data_out = train.loc[13023:13059].reset_index(drop = True)
    data_out['congestion_pred'] = rf_val_pred
    test['congestion_pred'] = rf_test_pred
    
    return [data_out[['row_id', 'time', 'x', 'y', 'direction', 'congestion', 'congestion_pred']], test[['row_id', 'time', 'x', 'y', 'direction', 'congestion_pred']]]