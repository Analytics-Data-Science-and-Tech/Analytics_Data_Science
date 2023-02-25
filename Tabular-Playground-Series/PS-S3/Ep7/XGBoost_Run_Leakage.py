import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt; plt.style.use('ggplot')
import seaborn as sns

from scipy.stats import rankdata
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, StratifiedKFold, TimeSeriesSplit
from sklearn.feature_selection import RFE, RFECV
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
# from lightgbm import xgbMClassifier
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

import optuna 

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Tabular-Playground-Series/PS-S3/Ep7/train.csv'
file_key_2 = 'Tabular-Playground-Series/PS-S3/Ep7/test.csv'
file_key_3 = 'Tabular-Playground-Series/PS-S3/Ep7/sample_submission.csv'

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

#########################
## Feature Engineering ##
#########################

## Fixing dates (https://www.kaggle.com/competitions/playground-series-s3e7/discussion/386655)
train['arrival_year_month'] = pd.to_datetime(train['arrival_year'].astype(str) + train['arrival_month'].astype(str), format = '%Y%m')
test['arrival_year_month'] = pd.to_datetime(test['arrival_year'].astype(str) + test['arrival_month'].astype(str), format = '%Y%m')

train.loc[train.arrival_date > train.arrival_year_month.dt.days_in_month, 'arrival_date'] = train.arrival_year_month.dt.days_in_month
test.loc[test.arrival_date > test.arrival_year_month.dt.days_in_month, 'arrival_date'] = test.arrival_year_month.dt.days_in_month

train.drop(columns = 'arrival_year_month', inplace = True)
test.drop(columns = 'arrival_year_month', inplace = True)

train['low_price_flag'] = np.where(train['avg_price_per_room'] < 30, 1, 0)
train['segment_0'] = np.where(train['market_segment_type'] == 0, 1, 0)
train['segment_1'] = np.where(train['market_segment_type'] == 1, 1, 0)
train['total_guests'] = train['no_of_adults'] + train['no_of_children']
train['stay_length'] = train['no_of_weekend_nights'] + train['no_of_week_nights']
train['stay_during_weekend'] = np.where(train['no_of_weekend_nights'] > 0, 1, 0)
train['quarter_1'] = np.where(train['arrival_month'] <= 3, 1, 0)
train['quarter_2'] = np.where(((train['arrival_month'] >= 4) & (train['arrival_month'] <= 6)), 1, 0)
train['quarter_3'] = np.where(((train['arrival_month'] >= 7) & (train['arrival_month'] <= 9)), 1, 0)
train['quarter_4'] = np.where(train['arrival_month'] >= 10, 1, 0)
train['segment_0_feature_1'] = np.where(((train['market_segment_type'] == 0) & (train['lead_time'] <= 90)), 1, 0)
train['segment_0_feature_2'] = np.where(((train['market_segment_type'] == 0) & (train['avg_price_per_room'] > 98)), 1, 0)
train['segment_1_feature_1'] = np.where(((train['market_segment_type'] == 1) & (train['no_of_special_requests'] == 0)), 1, 0)
train['segment_1_feature_2'] = np.where(((train['market_segment_type'] == 1) & (train['no_of_special_requests'] > 0) & (train['lead_time'] <= 150)), 1, 0)
train['segment_0_year_flag'] = np.where(((train['market_segment_type'] == 0) & (train['arrival_year'] == 2018)), 1, 0)
train['segment_1_year_flag'] = np.where(((train['market_segment_type'] == 1) & (train['arrival_year'] == 2018)), 1, 0)
train['price_lead_time_flag'] = np.where(((train['avg_price_per_room'] > 100) & (train['lead_time'] > 150)), 1, 0)

test['low_price_flag'] = np.where(test['avg_price_per_room'] < 30, 1, 0)
test['segment_0'] = np.where(test['market_segment_type'] == 0, 1, 0)
test['segment_1'] = np.where(test['market_segment_type'] == 1, 1, 0)
test['total_guests'] = test['no_of_adults'] + test['no_of_children']
test['stay_length'] = test['no_of_weekend_nights'] + test['no_of_week_nights']
test['stay_during_weekend'] = np.where(test['no_of_weekend_nights'] > 0, 1, 0)
test['quarter_1'] = np.where(test['arrival_month'] <= 3, 1, 0)
test['quarter_2'] = np.where(((test['arrival_month'] >= 4) & (test['arrival_month'] <= 6)), 1, 0)
test['quarter_3'] = np.where(((test['arrival_month'] >= 7) & (test['arrival_month'] <= 9)), 1, 0)
test['quarter_4'] = np.where(test['arrival_month'] >= 10, 1, 0)
test['segment_0_feature_1'] = np.where(((test['market_segment_type'] == 0) & (test['lead_time'] <= 90)), 1, 0)
test['segment_0_feature_2'] = np.where(((test['market_segment_type'] == 0) & (test['avg_price_per_room'] > 98)), 1, 0)
test['segment_1_feature_1'] = np.where(((test['market_segment_type'] == 1) & (test['no_of_special_requests'] == 0)), 1, 0)
test['segment_1_feature_2'] = np.where(((test['market_segment_type'] == 1) & (test['no_of_special_requests'] > 0) & (test['lead_time'] <= 150)), 1, 0)
test['segment_0_year_flag'] = np.where(((test['market_segment_type'] == 0) & (test['arrival_year'] == 2018)), 1, 0)
test['segment_1_year_flag'] = np.where(((test['market_segment_type'] == 1) & (test['arrival_year'] == 2018)), 1, 0)
test['price_lead_time_flag'] = np.where(((test['avg_price_per_room'] > 100) & (test['lead_time'] > 150)), 1, 0)


##########################
## Splitting Duplicates ##
##########################

train_dup = train.copy()
test_dup = test.copy()

duplicates = pd.merge(train, test, on = train_dup.columns.tolist()[1:18])
train_dup_ids = duplicates['id_x'].tolist()
test_dup_ids = duplicates['id_y'].tolist()

train_clean = train[~np.isin(train['id'], train_dup_ids)].reset_index(drop = True)
train_dup = train[np.isin(train['id'], train_dup_ids)].reset_index(drop = True)

test_clean = test[~np.isin(test['id'], test_dup_ids)].reset_index(drop = True)
test_dup = test[np.isin(test['id'], test_dup_ids)].reset_index(drop = True)


#######################
## Feature Selection ##
#######################

print('-----------------------------------')
print(' (-: Feature Selection Started :-) ')
print('-----------------------------------')


X = train_clean.drop(columns = ['id', 'low_price_flag', 'no_of_adults', 'no_of_children', 'no_of_weekend_nights', 'no_of_week_nights', 'booking_status'], axis = 1)
Y = train_clean['booking_status'] 

## Running RFECV multiple times
RFE_results = list()

for i in tqdm(range(0, 10)):
    
    auto_feature_selection = RFECV(estimator = XGBClassifier(), step = 1, min_features_to_select = 2, cv = 5, scoring = 'roc_auc', n_jobs = -1).fit(X, Y)
    
    ## Extracting and storing features to be selected
    RFE_results.append(auto_feature_selection.support_)

## Changing to data-frame
RFE_results = pd.DataFrame(RFE_results)
RFE_results.columns = X.columns

## Computing the percentage of time features are flagged as important
RFE_results = 100*RFE_results.apply(np.sum, axis = 0) / RFE_results.shape[0]

## Identifying features with a percentage score > 80%
features_to_select = RFE_results.index[RFE_results > 80].tolist()
print(features_to_select)

############
## Optuna ##
############

print('-------------------------------------')
print(' (-: Optuna Optimization Started :-) ')
print('-------------------------------------')


X = train_clean[features_to_select]
Y = train_clean['booking_status']

test_xgb = test_clean[features_to_select]

class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        ## Parameters to be evaluated
        param = dict(objective = 'binary:logistic',
                     eval_metric = 'auc',
                     tree_method = 'hist', 
                     max_depth = trial.suggest_int('max_depth', 2, 10),
                     learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log = True),
                     n_estimators = trial.suggest_int('n_estimators', 30, 10000),
                     gamma = trial.suggest_float('gamma', 0, 10),
                     min_child_weight = trial.suggest_int('min_child_weight', 1, 100),
                     colsample_bytree = trial.suggest_float('colsample_bytree', 0.2, 0.9),
                     subsample = trial.suggest_float('subsample', 0.2, 0.9)
                    )

        scores = []
        
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.seed)

        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = XGBClassifier(**param).fit(X_train, Y_train)
            preds_valid = model.predict_proba(X_valid)[:, 1]

            score = roc_auc_score(Y_valid, preds_valid)
            scores.append(score)
            
        return np.mean(scores)
    
## Defining number of runs and seed
# RUNS = 50
SEED = 42
N_TRIALS = 60

# Execute an optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)

print('----------------------------------------')
print(' (-: Saving Optuna Hyper-Parameters :-) ')
print('----------------------------------------')

optuna_hyper_params = pd.DataFrame.from_dict([study.best_trial.params])
file_name = 'XGB_FS_Seed_' + str(SEED) + '_Optuna_Hyperparameters.csv'
optuna_hyper_params.to_csv(file_name, index = False)

print('-----------------------------')
print(' (-: Starting CV process :-) ')
print('-----------------------------')

XGB_cv_scores, roc_auc_scores = list(), list()
preds = list()

## Running 5 times CV
for i in tqdm(range(5)):
    
    skf = StratifiedKFold(n_splits = 5, random_state = SEED, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building RF model
        XGB_md = XGBClassifier(**study.best_trial.params).fit(X_train, Y_train)
#         XGB_md = XGBClassifier().fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict_proba(X_test)[:, 1]
        XGB_pred_2 = XGB_md.predict_proba(test_xgb)[:, 1]
        
        ## Computing roc-auc score
        roc_auc_scores.append(roc_auc_score(Y_test, XGB_pred_1))
        preds.append(XGB_pred_2)

    XGB_cv_scores.append(np.mean(roc_auc_scores))
    
XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average oof roc-auc score over 5-folds (run 5 times) is:', XGB_cv_score)

###############################
## Consolidating Predictions ##
###############################

xgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)
clean_pred = pd.DataFrame({'id': test_clean['id']})
clean_pred['booking_status_clean'] = xgb_preds_test

dup_pred = duplicates[['id_y', 'booking_status']]
dup_pred.columns = ['id', 'booking_status_dup']
dup_pred['booking_status_dup'] = 1 - dup_pred['booking_status_dup']

submission = pd.merge(submission.drop(columns = 'booking_status', axis = 1), clean_pred, on = 'id', how = 'left')
submission = pd.merge(submission, dup_pred, on = 'id', how = 'left')
submission['booking_status'] = np.where(np.isnan(submission['booking_status_clean']), submission['booking_status_dup'], submission['booking_status_clean'])
submission.drop(columns = ['booking_status_clean', 'booking_status_dup'], axis = 1, inplace = True)

file_name = 'XGBoost_Leakage_2_' + str(SEED) + '_.csv'
submission.to_csv(file_name, index = False)
# CV_scores.to_csv(file_name, index = False)

print('--------------------------')    
print('...The process finished...')    
print('--------------------------')