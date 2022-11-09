import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

from Run_LightGBM_Best_Help import Run_LightGBM_Best

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

CV_scores = list()    
    
for i in range(0, 500):
    
    print('Working in', i, ' Run')
    run = Run_LightGBM(X, Y, test_new, submission)
    CV_scores.append(run[0])
    location_name = 's3://analytics-data-science-competitions/Tabular-Playground-Series/Tabular-Playground-Nov-2022/LightGBM_Preds/' + 'LightGBM_run_' + str(i) + '.csv'
    run[1].to_csv(location_name, index = False)
    
    
CV_scores = pd.DataFrame({'Run': [i for i in range(0, 500)], 'CV_score': CV_scores})
CV_scores.to_csv('s3://analytics-data-science-competitions/Tabular-Playground-Series/Tabular-Playground-Nov-2022/LightGBM_Preds/CV_scores.csv', index = False)

