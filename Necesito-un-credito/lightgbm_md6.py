import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from lightgbm import LGBMClassifier
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
LightGBM_param_grid = {'n_estimators': [300],
                       'max_depth': [7, 9],
                       'num_leaves': [50, 60],
                       'min_data_in_leaf': [30, 35],
                       'learning_rate': [0.01, 0.001],
                       'feature_fraction': [0.4, 0.5, 0.6],
                       'lambda_l1': [0],
                       'lambda_l2': [0]
                      }

## Performing grid search with 5 folds
LightGBM_grid_search = GridSearchCV(LGBMClassifier(), LightGBM_param_grid, cv = 5, scoring = 'roc_auc', n_jobs = -1, verbose = 3).fit(X, Y)

## Extracting best hyper-parameters
best_params = LightGBM_grid_search.best_params_
print('The best hyper-parameters are:', best_params)

## Extracting the best score
best_score = LightGBM_grid_search.best_score_
print('The best area under the ROC cure is:', best_score)

## Extracting the best model
LightGBM_md = LightGBM_grid_search.best_estimator_

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
lightgbm_pred_train = LightGBM_md.predict_proba(X)[:, 1]
opt_cutoff = roc_auc_cutoff(Y, lightgbm_pred_train)
print('The optimal cutoff is', opt_cutoff)

## Predicting on the test
lightgbm_pred_test = LightGBM_md.predict_proba(test)[:, 1]
lightgbm_label_test = np.where(lightgbm_pred_test < opt_cutoff, 0, 1)

## Data-frame for submission
data_out = pd.DataFrame({'Id': test_id, 'SeriousDlqin2yrs': lightgbm_label_test})
data_out.to_csv('lightgbm_submission_md6.csv', index = False)

# The best hyper-parameters are: {'feature_fraction': 0.5, 'lambda_l1': 0, 'lambda_l2': 0, 'learning_rate': 0.01, 'max_depth': 7, 'min_data_in_leaf': 30, 'n_estimators': 300, 'num_leaves': 50}

# The best area under the ROC cure is: 0.8628911985195646
# The optimal cutoff is 0.06901024161451128
