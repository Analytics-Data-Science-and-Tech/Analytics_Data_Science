import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve
from sklearn.impute import KNNImputer

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Necesito-un-credito/train.csv'
file_key_2 = 'Necesito-un-credito/test.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

## Reading data-files
train = pd.read_csv(file_content_stream_1)
train['age'] = [train['age'][i][:-1] for i in range(0, train.shape[0])]
train['age'] = pd.to_numeric(train['age'])

test = pd.read_csv(file_content_stream_2)
test['age'] = [test['age'][i][:-1] for i in range(0, test.shape[0])]
test['age'] = pd.to_numeric(test['age'])

test_id = test['Id']
test = test.drop(columns = ['Id'], axis = 1)

## Defining input and target variables
X = train.drop(columns = ['Id', 'SeriousDlqin2yrs'], axis = 1)
Y = train['SeriousDlqin2yrs']

## Defining the hyper-parameter grid
XGBoost_param_grid = {'n_estimators': [300],
                      'max_depth': [7, 9],
                      'min_child_weight': [10, 15, 20],
                      'learning_rate': [0.01, 0.001],
                      'gamma': [0.3, 0.1],
                      'subsample': [0.8, 1],
                      'colsample_bytree': [0.8, 1]}

## Performing grid search with 5 folds
XGBoost_grid_search = GridSearchCV(XGBClassifier(), XGBoost_param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1, verbose = 3).fit(X, Y)

## Extracting best hyper-parameters
best_params = XGBoost_grid_search.best_params_
print('The best hyper-parameters are:', best_params)

## Extracting the best score
best_score = XGBoost_grid_search.best_score_
print('The best area under the ROC cure is:', best_score)

## Extracting the best model
XGBoost_md = XGBoost_grid_search.best_estimator_

def roc_auc_cutoff(Y_test, Y_pred):
    
    ## Computing the precision recall curve
    fpr, tpr, thresholds = roc_curve(Y_test, Y_pred)
    
    cutoffs = pd.DataFrame({'False_Positive': fpr, 'True_Positive': tpr, 'cutoff': thresholds})

    ## Finding the optimal cut-off
    cutoffs['True_Positive_minus_1'] = cutoffs['True_Positive'] - 1
    cutoffs['Euclidean_dist'] = np.sqrt(cutoffs['False_Positive']**2 + cutoffs['True_Positive_minus_1']**2)

    ## Sorting based on the Euclidean distance
    cutoffs = cutoffs.sort_values(by = 'Euclidean_dist').reset_index(drop = True)
        
    return cutoffs['cutoff'][0]

## Predicting on train to estimate cutoff
xgb_pred_train = XGBoost_md.predict_proba(X)[:, 1]
opt_cutoff = roc_auc_cutoff(Y, xgb_pred_train)
print('The optimal cutoff is', opt_cutoff)

## Predicting on the test
xgb_pred_test = XGBoost_md.predict_proba(test)[:, 1]
xgb_label_test = np.where(xgb_pred_test < opt_cutoff, 0, 1)

## Data-frame for submission
data_out = pd.DataFrame({'Id': test_id, 'SeriousDlqin2yrs': xgb_label_test})
data_out.to_csv('xgb_submission_md1.csv', index = False)


