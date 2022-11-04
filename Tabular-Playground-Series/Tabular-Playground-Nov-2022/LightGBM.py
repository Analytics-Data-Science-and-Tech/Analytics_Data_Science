import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Tabular-Playground-Series/Tabular-Playground-Nov-2022/sample_submission.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

## Reading data-files
submission = pd.read_csv(file_content_stream_1)
df = pd.read_parquet('s3://analytics-data-science-competitions/Tabular-Playground-Series/Tabular-Playground-Nov-2022/preds_concat_gzip.parquet', engine = 'fastparquet')

## train and test
preds_df = df.clip(0, 1) 
train = preds_df[preds_df['target'].notnull()]
test = preds_df[preds_df['target'].isnull()] 

## Reading logloss data 
logloss_data = pd.read_csv('scores_df.csv')
logloss_data = logloss_data.sort_values(by = 'split_score', ascending = False)
# logloss_data = logloss_data.sort_values(by = 'gain_score', ascending = False)
# logloss_data = pd.read_csv('logloss_data.csv')

# X = train[logloss_data['File'][0:100].values]
# Y = train['target']

# test_new = test[logloss_data['File'][0:100].values]

X = train[logloss_data['feature'][0:300].values] 
Y = train['target']

test_new = test[logloss_data['feature'][0:300].values] 


## Defining list to store results
lgb_results, test_preds_lgb = list(), list()

fold = 1
kfold = StratifiedKFold(n_splits = 5, shuffle = True)
        
for train_ix, test_ix in kfold.split(X, Y):
    
    ## Splitting the data 
    X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
    Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]

    ## Building model
    lgb_md = LGBMClassifier(n_estimators = 1000, 
                            learning_rate = 0.01,
                            num_leaves = 50,
                            max_depth = 17, 
                            lambda_l1 = 3, 
                            lambda_l2 = 1, 
                            bagging_fraction = 0.95, 
                            feature_fraction = 0.96).fit(X_train, Y_train)
        
    ## Predicting on test
    lgb_pred = lgb_md.predict_proba(X_test)[:, 1]
    score = log_loss(Y_test, lgb_pred)
    lgb_results.append(score)
        
    print('Fold ', str(fold), ' result is:', score, '\n')

    test_preds_lgb.append(lgb_md.predict_proba(test_new)[:, 1])
    fold +=1

print('The average log-loss over 5-fold CV is', np.mean(lgb_results))

test_preds_lgb = pd.DataFrame(test_preds_lgb)
print(test_preds_lgb.shape)

test_preds_lgb = test_preds_lgb.mean(axis = 0)
print(test_preds_lgb.head(5))

submission['pred'] = test_preds_lgb
print(submission.head())

submission.to_csv('submission_LightGBM_300_split_1.csv', index = False)