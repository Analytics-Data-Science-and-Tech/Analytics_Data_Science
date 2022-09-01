import boto3
import pandas as pd 
import numpy as np
from Running_XGB_Help_Funs import main_XGB_all_directions

## Defining the bucket 
s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

## Defining the csv file 
file_key_train = 'Forecasting-Flow-Traffic/train.csv'
file_key_test = 'Forecasting-Flow-Traffic/test.csv'

bucket_object_train = bucket.Object(file_key_train)
file_object_train = bucket_object_train.get()
file_content_stream_train = file_object_train.get('Body')

bucket_object_test = bucket.Object(file_key_test)
file_object_test = bucket_object_test.get()
file_content_stream_test = file_object_test.get('Body')

## Reading the csv file
train = pd.read_csv(file_content_stream_train)
test = pd.read_csv(file_content_stream_test)

## Puting time in the right format 
train['time'] = pd.to_datetime(train['time'], format = '%Y-%m-%d %H:%M:%S')
test['time'] = pd.to_datetime(test['time'], format = '%Y-%m-%d %H:%M:%S')

## Extracting day, hour and minute
train['day'] = train['time'].dt.dayofweek
train['hour'] = train['time'].dt.hour
train['minute'] = train['time'].dt.minute

test['day'] = test['time'].dt.dayofweek
test['hour'] = test['time'].dt.hour
test['minute'] = test['time'].dt.minute

## Changing direction to dummies
train = pd.concat([train, pd.get_dummies(train['direction'])], axis = 1)
test = pd.concat([test, pd.get_dummies(train['direction'])], axis = 1)

## Modeling 
results = main_XGB_all_directions(train, test)

## Storing results
results[0].to_csv('results_validation.csv', index = False)
results[1].to_csv('results_test.csv', index = False)
