import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

from tqdm import tqdm

from functools import partial
import scipy as sp

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns

from scipy.stats import rankdata
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor 
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import optuna 

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Tabular-Playground-Series/PS-S3/Ep9/train.csv'
file_key_2 = 'Tabular-Playground-Series/PS-S3/Ep9/test.csv'
file_key_3 = 'Tabular-Playground-Series/PS-S3/Ep9/sample_submission.csv'

bucket_object_1 = bucket.Object(file_key_1)
file_object_1 = bucket_object_1.get()
file_content_stream_1 = file_object_1.get('Body')

bucket_object_2 = bucket.Object(file_key_2)
file_object_2 = bucket_object_2.get()
file_content_stream_2 = file_object_2.get('Body')

bucket_object_3 = bucket.Object(file_key_3)
file_object_3 = bucket_object_3.get()
file_content_stream_3 = file_object_3.get('Body')

## Reading data files
train = pd.read_csv(file_content_stream_1)
test = pd.read_csv(file_content_stream_2)
submission = pd.read_csv(file_content_stream_3)

X = train.drop(columns = ['id', 'Strength'], axis = 1)
Y = train['Strength']
X['WaterComponent_to_Cement_ratio'] = X['WaterComponent'] / (X['CementComponent'] + 1e-6)

test_baseline = test.drop(columns = ['id'], axis = 1)
test_baseline['WaterComponent_to_Cement_ratio'] = test_baseline['WaterComponent'] / (test_baseline['CementComponent'] + 1e-6)

hist_md = HistGradientBoostingRegressor(l2_regularization = 0.01,
                                        early_stopping = False,
                                        learning_rate = 0.01,
                                        max_iter = 1000,
                                        max_depth = 2,
                                        max_bins = 255,
                                        min_samples_leaf = 10,
                                        max_leaf_nodes = 10).fit(X, Y)

XGB_md = XGBRegressor(tree_method = 'hist',
                      colsample_bytree = 0.7, 
                      gamma = 0.8, 
                      learning_rate = 0.01, 
                      max_depth = 2, 
                      min_child_weight = 10, 
                      n_estimators = 1000, 
                      subsample = 0.7).fit(X, Y)

xgb_pred = XGB_md.predict(test_baseline)

cat_md = CatBoostRegressor(loss_function = 'RMSE',
                           iterations = 1000,
                           learning_rate = 0.01,
                           depth = 3,
                           random_strength = 0.5,
                           bagging_temperature = 0.7,
                           border_count = 30,
                           l2_leaf_reg = 5,
                           verbose = False).fit(X, Y)

cat_pred = cat_md.predict(test_baseline)

submission['Strength'] = (cat_pred + xgb_pred) / 2
submission.head()