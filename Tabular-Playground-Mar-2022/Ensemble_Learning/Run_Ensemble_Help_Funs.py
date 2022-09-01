import pandas as pd 
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def Run_Ensemble(train, test, model):
    
    """
    This function loops through all the directions and locations.
    With the goal of ensemble the prediction from the four
    different models that were built. It takes three arguments:
    train: this is the train data-frame.
    test: this is the test data-frame. 
    model: model to be used to ensemble the predictions 
    """
    
    ## Defining all the directions
    directions = train['direction'].unique()
    
    ## Defining lists to store results
    results_all_directions_val = list()
    results_all_directions_test = list()
    
    for i in range(0, len(directions)):
        print('Working on direction:', directions[i])
        ## Subsetting train & test based on directions
        temp_train = train[train['direction'] == directions[i]].reset_index(drop = True)
        temp_test = test[test['direction'] == directions[i]].reset_index(drop = True)
        
        ## Appending results 
        results = Run_Ensemble_help(temp_train, temp_test, model)
        results_all_directions_val.append(results[0])
        results_all_directions_test.append(results[1])
            
    return [pd.concat(results_all_directions_val), pd.concat(results_all_directions_test)]
    
    
def Run_Ensemble_help(train, test, model):
    
    ## Defining lists to store results
    results_all_locations_val = list()
    results_all_locations_test = list()
    
    ## Defining locations 
    x_values = train['x'].unique()
    y_values = train['y'].unique()
    
    ## Defining list to store results
    results_all_locations = list()
    
    for i in range(0, len(x_values)):
        
        for j in range(0, len(y_values)):
            print('location: (',x_values[i],',',y_values[j],')')
            ## Subsetting train & test based on locaitons
            temp_train = train[(train['x'] == x_values[i]) & (train['y'] == y_values[j])].reset_index(drop = True)
            temp_test = test[(test['x'] == x_values[i]) & (test['y'] == y_values[j])].reset_index(drop = True)
            
            ## Sanity check
            if (temp_train.shape[0] == 0):
                
                continue
            
            ## Modeling building and prediction at location (x, y)
            results = Run_Ensemble_help_help(temp_train, temp_test, model)
            results_all_locations_val.append(results[0])
            results_all_locations_test.append(results[1])
            
    return [pd.concat(results_all_locations_val), pd.concat(results_all_locations_test)]


def Run_Ensemble_help_help(train, test, model):
    
    """
    This function conducts leave-one-out cross validation to 
    tuned the model hyper-parameters. It takes three arguments
    train: train dataset
    test: test dataset 
    model: model to be considered
    """
    
    ## Defining the input & targets
    X_train = train[['congestion_pred_1', 'congestion_pred_2', 'congestion_pred_3', 'congestion_pred_4']]
    X_test = test[['congestion_pred_1', 'congestion_pred_2', 'congestion_pred_3', 'congestion_pred_4']]
    Y_train = train['congestion']
    
    
    if (model == 'RF'):
        
        ###############################################
        ## Defining hyer-parameters to be considered ##
        ###############################################

        ## Number of trees in random forest
        n_estimators = [100, 300, 500]

        ## Number of features to consider at every split
        max_features = [3, 4]

        ## Maximum number of levels in tree
        max_depth = [3, 5]

        ## Minimum number of samples required to split a node
        min_samples_split = [5, 7]

        ## Minimum number of samples required at each leaf node
        min_samples_leaf = [3, 5]

        ## Creating the dictionary of hyper-parameters
        RF_param_grid = {'n_estimators': n_estimators,
                         'max_features': max_features,
                         'max_depth': max_depth,
                         'min_samples_split': min_samples_split,
                         'min_samples_leaf': min_samples_leaf}
        
        ## Running leave-one-out cross validation 
        RF_grid_search = GridSearchCV(RandomForestRegressor(), RF_param_grid, cv = LeaveOneOut(), scoring = 'neg_mean_squared_error', n_jobs = -1).fit(X_train, Y_train)

        ## Extraciting the best model 
        RF_md = RF_grid_search.best_estimator_
        
        ## Predicting on validation and test
        RF_val_pred = RF_md.predict(X_train)
        RF_test_pred = RF_md.predict(X_test)
        
        ## Appending results 
        train['congestion_ensemble_pred'] = RF_val_pred
        test['congestion_ensemble_pred'] = RF_test_pred
        
        return [train, test]
    
    
    if (model == 'svm'):
        
        ###############################################
        ## Defining hyer-parameters to be considered ##
        ###############################################

        ## Kernel
        kernel = ['rbf', 'poly', 'sigmoid']

        ## Regularization parameter
        C = [0.01, 0.1, 1, 10]

        ## Gamma
        gamma = [0.001, 0.01, 0.1, 1]

        ## Creating the dictionary of hyper-parameters
        SVM_param_grid = {'kernel': kernel,
                          'C': C,
                          'gamma': gamma}
        
        ## Running leave-one-out cross validation 
        svm_grid_search = GridSearchCV(SVR(), SVM_param_grid, cv = LeaveOneOut(), scoring = 'neg_mean_squared_error', n_jobs = -1).fit(X_train, Y_train)

        ## Extraciting the best model 
        svm_md = svm_grid_search.best_estimator_
        
        ## Predicting on validation and test
        svm_val_pred = svm_md.predict(X_train)
        svm_test_pred = svm_md.predict(X_test)
        
        ## Appending results 
        train['congestion_ensemble_pred'] = svm_val_pred
        test['congestion_ensemble_pred'] = svm_test_pred
        
        return [train, test]
