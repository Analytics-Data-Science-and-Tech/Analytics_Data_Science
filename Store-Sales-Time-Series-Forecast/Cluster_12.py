import boto3
import pandas as pd
import numpy as np

import time

from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold

from lightgbm import LGBMRegressor

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining files names
file_key_1 = 'Store-Sales-Time-Series-Forecast/oil.csv'
file_key_2 = 'Store-Sales-Time-Series-Forecast/holidays_events.csv'
file_key_3 = 'Store-Sales-Time-Series-Forecast/stores.csv'
file_key_4 = 'Store-Sales-Time-Series-Forecast/transactions.csv'
file_key_5 = 'Store-Sales-Time-Series-Forecast/train.csv'
file_key_6 = 'Store-Sales-Time-Series-Forecast/test.csv'
file_key_7 = 'Store-Sales-Time-Series-Forecast/sample_submission.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

bucket_object_3 = bucket.Object(file_key_3)
file_object_3 = bucket_object_3.get()
file_content_stream_3 = file_object_3.get('Body')

bucket_object_4 = bucket.Object(file_key_4)
file_object_4 = bucket_object_4.get()
file_content_stream_4 = file_object_4.get('Body')

bucket_object_5 = bucket.Object(file_key_5)
file_object_5 = bucket_object_5.get()
file_content_stream_5 = file_object_5.get('Body')

bucket_object_6 = bucket.Object(file_key_6)
file_object_6 = bucket_object_6.get()
file_content_stream_6 = file_object_6.get('Body')

bucket_object_7 = bucket.Object(file_key_7)
file_object_7 = bucket_object_7.get()
file_content_stream_7 = file_object_7.get('Body')

## Reading data-files
oil = pd.read_csv(file_content_stream_1)
holidays = pd.read_csv(file_content_stream_2)
stores = pd.read_csv(file_content_stream_3)
transactions = pd.read_csv(file_content_stream_4)
train = pd.read_csv(file_content_stream_5)
test = pd.read_csv(file_content_stream_6)
submission = pd.read_csv(file_content_stream_7)

## Updating holiday column names
holidays.columns = ['date', 'holiday_type', 'locale', 'locale_name', 'description', 'transferred']

## Updating store column names
stores.columns = ['store_nbr', 'city', 'state', 'store_type', 'cluster']

###################
## Train Dataset ##
###################

train = pd.merge(train, oil, on = 'date', how = 'left')
train = pd.merge(train, holidays, on = 'date', how = 'left')
train = pd.merge(train, stores, on = 'store_nbr', how = 'left')
train['date'] = pd.to_datetime(train['date'], format = '%Y-%m-%d')

## Basic feature engineering 
cluster_dummies = pd.get_dummies(train['cluster'])
cluster_dummies.columns = ['cluster_' + str(i) for i in range(1, 18)]
train = pd.concat([train.drop(columns = ['cluster'], axis = 1), cluster_dummies], axis = 1)

family_dummies = pd.get_dummies(train['family'])
family_dummies.columns = ['family_' + str(i) for i in range(1, 34)]
train = pd.concat([train.drop(columns = ['family'], axis = 1), family_dummies], axis = 1)

train['day'] = train['date'].dt.dayofweek
train['month'] = train['date'].dt.month
# train['year'] = train['date'].dt.year
train['is_holiday'] = np.where(train['holiday_type'] == 'Holiday', 1, 0)

##################
## Test Dataset ##
##################

## Appending oil prices and holiday
test = pd.merge(test, holidays, on = 'date', how = 'left')
test = pd.merge(test, oil, on = 'date', how = 'left')
test = pd.merge(test, stores, on = 'store_nbr', how = 'left')
test['date'] = pd.to_datetime(test['date'], format = '%Y-%m-%d')

## Basic feauture engineering 
cluster_dummies = pd.get_dummies(test['cluster'])
cluster_dummies.columns = ['cluster_' + str(i) for i in range(1, 18)]
test = pd.concat([test.drop(columns = ['cluster'], axis = 1), cluster_dummies], axis = 1)

family_dummies = pd.get_dummies(test['family'])
family_dummies.columns = ['family_' + str(i) for i in range(1, 34)]
test = pd.concat([test.drop(columns = ['family'], axis = 1), family_dummies], axis = 1)

test['day'] = test['date'].dt.dayofweek
test['month'] = test['date'].dt.month
# test['year'] = test['date'].dt.year
test['is_holiday'] = np.where(test['holiday_type'] == 'Holiday', 1, 0)

###############
## Cluster 1 ##
###############

train = train[train['cluster_11'] == 1].reset_index(drop = True)
test = test[test['cluster_11'] == 1].reset_index(drop = True)

X = train.drop(columns = ['id', 'date', 'store_nbr', 'sales', 'holiday_type', 'locale', 'locale_name', 'description', 'transferred', 'city', 'state', 'store_type'], axis = 1)
Y = train['sales']

test_ids = test['id']
test = test.drop(columns = ['id', 'date', 'store_nbr', 'holiday_type', 'locale', 'locale_name', 'description', 'transferred', 'city', 'state', 'store_type'], axis = 1)

t1 = time.time()
# kf = GroupKFold(n_splits = 5)
kf = KFold(n_splits = 5, shuffle = True, random_state = 888)
score_list_lgb = []
test_preds_lgb = []
fold = 1

# for train_index, test_index in kf.split(X, Y, groups = X.year):
for train_index, test_index in kf.split(X, Y):
    
    ## Splitting the data
    X_train , X_val = X.iloc[train_index], X.iloc[test_index]  
    Y_train, Y_val = Y.iloc[train_index], Y.iloc[test_index]    
    
    print("X_train shape is :", X_train.shape, "X_val shape is", X_val.shape)
    y_pred_list = []
    
    model_lgb = LGBMRegressor(n_estimators = 5000, 
                              learning_rate = 0.01,
                              num_leaves = 50,
                              max_depth = 17, 
                              lambda_l1 = 3, 
                              lambda_l2 = 1, 
                              bagging_fraction = 0.9, 
                              feature_fraction = 0.9)

    model = model_lgb.fit(X_train, Y_train)
    result = model_lgb.predict(X_val)
    
    result = pd.DataFrame(result)
    result.iloc[:, 0] = [0 if i <= 0 else i for i in result.iloc[:,0]]
    
    score = np.sqrt(mean_squared_log_error(Y_val, result))
    print('Fold ', str(fold), ' result is:', score, '\n')
    score_list_lgb.append(score)
    
    test_preds = model_lgb.predict(test)
    test_preds = np.where(test_preds < 0, 0, test_preds)
    test_preds_lgb.append(test_preds)
    fold = fold + 1

t2 = time.time()
print('-- Model Summary Results --', '\n')
print("LGBM model with cross validation take : {:.3f} sn.".format(t2-t1), '\n')

mean = sum(score_list_lgb) / len(score_list_lgb)
variance = sum([((x - mean) ** 2) for x in score_list_lgb]) / len(score_list_lgb)
res = variance ** 0.5

print("Cross validation mean score:", sum(score_list_lgb) / len(score_list_lgb), '\n')
print("Cross validation score's Standart deviation is:", res, '\n')

test_preds_lgb = pd.DataFrame(test_preds_lgb)
test_preds_lgb = test_preds_lgb.mean(axis = 0)

data_out = pd.DataFrame({'id': test_ids})
data_out['sales'] = test_preds_lgb
data_out.to_csv('Cluster_12.csv', index = False)

print('-- Process Finished --')

