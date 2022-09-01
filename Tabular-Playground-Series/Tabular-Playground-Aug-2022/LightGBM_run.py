import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

import os
import sagemaker

sess = sagemaker.Session()

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Tabular-Playground-Aug-2022/train.csv'
file_key_2 = 'Tabular-Playground-Aug-2022/test.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

## Reading data-files
train = pd.read_csv(file_content_stream_1)
train = train.drop(columns = ['id'], axis = 1)

test = pd.read_csv(file_content_stream_2)
test_id = test['id']
test = test.drop(columns = ['id'], axis = 1)

## Changing labels to dummies
train_dummies = pd.get_dummies(train[['attribute_0']])
train = train.drop(columns = ['product_code', 'attribute_0', 'attribute_1'], axis = 1)
train = pd.concat([train, train_dummies], axis = 1)

test_dummies = pd.get_dummies(test[['attribute_0']])
test = test.drop(columns = ['product_code', 'attribute_0', 'attribute_1'], axis = 1)
test = pd.concat([test, test_dummies], axis = 1)

## Defining input and target variables
X = train.drop(columns = ['failure'], axis = 1)
Y = train['failure']

## Defining the hyper-parameter grid
LightGBM_param_grid = {'n_estimators': [100, 300],
                       'max_depth': [3, 5, 7],
                       'num_leaves': [20, 25, 30],
                       'min_data_in_leaf': [10, 15, 20],
                       'learning_rate': [0.01, 0.001],
                       'feature_fraction': [0.8, 0.9, 1],
                       'lambda_l1': [0, 10, 100],
                       'lambda_l2': [0, 10, 100]
                      }

## Performing grid search with 5 folds
LightGBM_grid_search = GridSearchCV(LGBMClassifier(), LightGBM_param_grid, cv = 3, scoring = 'roc_auc', n_jobs = -1).fit(X, Y)

## Extracting the best model
LightGBM_md = LightGBM_grid_search.best_estimator_

## Predicting on test with best LightGBM model 
lightGBM_pred = LightGBM_md.predict_proba(test)[:, 1] 

## Defining data-frame to be exported
data_out = pd.DataFrame({'id': test_id, 'failure': lightGBM_pred})
data_out.to_csv('submission.csv', index = False)

sess.upload_data(path = 'submission.csv', 
                 bucket = bucket_name,
                 key_prefix = 'Tabular-Playground-Aug-2022')