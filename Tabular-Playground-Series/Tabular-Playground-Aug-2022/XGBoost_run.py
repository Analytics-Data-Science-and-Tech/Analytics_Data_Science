import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.impute import KNNImputer
from xgboost import XGBClassifier

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

## Filling missing values with kNN
knn_imputer = KNNImputer(n_neighbors = 5, weights = 'distance')
train = pd.DataFrame(knn_imputer.fit_transform(train), columns = train.columns)
test = pd.DataFrame(knn_imputer.fit_transform(test), columns = test.columns)

## Engineering features
train['feature_1'] = np.where(train['loading'] < 150, 0, 1)
test['feature_1'] = np.where(test['loading'] < 150, 0, 1)

## Defining input and target variables
X = train.drop(columns = ['failure'], axis = 1)
Y = train['failure']

## Defining the hyper-parameter grid
XGBoost_param_grid = {'n_estimators': [300],
                      'max_depth': [5, 7],
                      'min_child_weight': [5, 7, 10],
                      'learning_rate': [0.01, 0.001],
                      'gamma': [0.3, 0.1],
                      'subsample': [0.8, 1],
                      'colsample_bytree': [0.8, 1]}

## Performing grid search with 5 folds
XGBoost_grid_search = GridSearchCV(XGBClassifier(), XGBoost_param_grid, cv = 3, scoring = 'roc_auc', n_jobs = -1).fit(X, Y)

## Extracting the best score
best_score = XGBoost_grid_search.best_score_
print('The best area under the ROC cure is:', best_score)

## Extracting the best model
XGBoost_md = XGBoost_grid_search.best_estimator_

## Predicting on test with best xgboost model 
xgb_pred = XGBoost_md.predict_proba(test)[:, 1] 

## Defining data-frame to be exported
data_out = pd.DataFrame({'id': test_id, 'failure': xgb_pred})
data_out.to_csv('XGB_submission.csv', index = False)

sess.upload_data(path = 'XGB_submission.csv', 
                 bucket = bucket_name,
                 key_prefix = 'Tabular-Playground-Aug-2022')