import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Tabular-Playground-Series/Tabular-Playground-Nov-2022/sample_submission.csv'
file_key_2 = 'Tabular-Playground-Series/Tabular-Playground-Nov-2022/train_labels.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

########################
## Reading data-files ##
########################

submission = pd.read_csv(file_content_stream_1)
y_true = pd.read_csv(file_content_stream_2)
df = pd.read_parquet('s3://analytics-data-science-competitions/Tabular-Playground-Series/Tabular-Playground-Nov-2022/preds_logit_concat_gzip.parquet', engine = 'fastparquet')
lasso_scores = pd.read_csv('lasso_scores_logit.csv')
to_select = lasso_scores['Feature'][0:50]

############################
## Consolidating the data ##
############################

preds = pd.merge(df, y_true, on = 'id', how = 'left')

############################
## train and test datsets ##
############################

train = preds[preds['label'].notnull()]
train['label'] = train['label'].astype(int)

test = preds[preds['label'].isnull()] 

##############
## Modeling ##
##############

X = train[to_select.values]
Y = train['label']

test_new = test[to_select.values]

## Defining list to store results
lgb_results = list()
test_preds_lgb_fold_1 = list() 
test_preds_lgb_fold_2 = list()
test_preds_lgb_fold_3 = list()
test_preds_lgb_fold_4 = list()
test_preds_lgb_fold_5 = list()

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
                            bagging_fraction = 0.4, 
                            feature_fraction = 0.4).fit(X_train, Y_train)
        
    ## Predicting on test
    lgb_pred = lgb_md.predict_proba(X_test)[:, 1]
    score = log_loss(Y_test, lgb_pred)
    lgb_results.append(score)
        
    print('Fold ', str(fold), ' result is:', score, '\n')
    
    if (fold == 1):
        test_preds_lgb_fold_1.append(lgb_md.predict_proba(test_new)[:, 1])
        
    if (fold == 2):
        test_preds_lgb_fold_2.append(lgb_md.predict_proba(test_new)[:, 1])
        
    if (fold == 3):
        test_preds_lgb_fold_3.append(lgb_md.predict_proba(test_new)[:, 1])
        
    if (fold == 4):
        test_preds_lgb_fold_4.append(lgb_md.predict_proba(test_new)[:, 1])
        
    if (fold == 5):
        test_preds_lgb_fold_5.append(lgb_md.predict_proba(test_new)[:, 1])
    
    fold +=1

print('The average log-loss over 5-fold CV is', np.mean(lgb_results))

##########################################
## Weighted average of fold predictions ##
##########################################

w1 = 1/ lgb_results[0]
w2 = 1/ lgb_results[1]
w3 = 1/ lgb_results[2]
w4 = 1/ lgb_results[3]
w5 = 1/ lgb_results[4]
w_tot = w1 + w2 + w3+ w4 + w5
w1 = w1 / w_tot
w2 = w2 / w_tot
w3 = w3 / w_tot
w4 = w4 / w_tot
w5 = w5 / w_tot

pred1 = w1*test_preds_lgb_fold_1[0]
pred2 = w2*test_preds_lgb_fold_2[0]
pred3 = w3*test_preds_lgb_fold_3[0]
pred4 = w4*test_preds_lgb_fold_4[0]
pred5 = w4*test_preds_lgb_fold_5[0]

submission['pred'] = pred1 + pred2 + pred3 + pred4 + pred5
submission.to_csv('s3://analytics-data-science-competitions/Tabular-Playground-Series/Tabular-Playground-Nov-2022/LightGBM_Preds/submission_1.csv', index = False)
                  


