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
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import optuna 

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Tabular-Playground-Series/PS-S3/Ep8/train.csv'
file_key_2 = 'Tabular-Playground-Series/PS-S3/Ep8/test.csv'
file_key_3 = 'Tabular-Playground-Series/PS-S3/Ep8/sample_submission.csv'

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

def updating_labels(df):
    
    df['clarity_scaled'] = df['clarity'].apply(lambda x: 0 if x == "IF" else 1 if x == "VVS1" else 2 if x == "VVS2" else 3 if x == "VS1" else 4 if x == "VS2" else 5 if x == "SI1" else 6 if x == "SI2" else 7)
    df['clarity_scaled'] = df['clarity_scaled'].astype(np.int8)
    
    df['cut_scaled'] = df['cut'].apply(lambda x: 0 if x == 'Fair' else 1 if x == 'Good' else 2 if x == 'Very Good' else 3 if x == 'Premium' else 4)                          
    df['cut_scaled'] = df['cut_scaled'].astype(np.int8) 
    
    df['color_scaled'] = df['color'].apply(lambda x: 0 if x == 'J' else 1 if x == 'I' else 2 if x == 'H' else 3 if x == 'G' else 4 if x == 'F' else 5 if x == 'E' else 6)
    df['color_scaled'] = df['color_scaled'].astype(np.int8)
    
    df.drop(columns = ['clarity', 'cut', 'color'], axis = 1, inplace = True)
    
    return df

train = updating_labels(train)
test = updating_labels(test)

##########################
## Splitting Duplicates ##
##########################

train_dup = train.copy()
test_dup = test.copy()

to_consider = ['carat', 'depth', 'table', 'x', 'y', 'z', 'clarity_scaled', 'cut_scaled', 'color_scaled']

duplicates = pd.merge(train, test, on = to_consider)
train_dup_ids = duplicates['id_x'].tolist()
test_dup_ids = duplicates['id_y'].tolist()

train_clean = train[~np.isin(train['id'], train_dup_ids)].reset_index(drop = True)
train_dup = train[np.isin(train['id'], train_dup_ids)].reset_index(drop = True)

test_clean = test[~np.isin(test['id'], test_dup_ids)].reset_index(drop = True)
test_dup = test[np.isin(test['id'], test_dup_ids)].reset_index(drop = True)

dup_pred_price = pd.DataFrame(train_dup.groupby(['clarity_scaled', 'cut_scaled', 'color_scaled'])['price'].mean()).reset_index()
test_dup = pd.merge(test_dup, dup_pred_price, on = ['clarity_scaled', 'cut_scaled', 'color_scaled'], how = 'left')
test_dup = test_dup[['id', 'price']]
test_dup.columns = ['id', 'price_dup']

############
## Optuna ##
############

print('------------------------------------')
print(' (-: Optuna Optimization Started :-)')
print('------------------------------------')

X = train_clean.drop(columns = ['id', 'price'], axis = 1)
Y = train_clean['price']

test_lgb = test_clean.drop(columns = 'id', axis = 1)

class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        
        ## Parameters to be evaluated
        param = dict(metric = 'rmse',
                     boosting_type = 'gbdt', 
                     n_estimators = trial.suggest_int('n_estimators', 300, 10000),
                     learning_rate = trial.suggest_float('learning_rate', 0.001, 1, log = True),
                     max_depth = trial.suggest_int('max_depth', 3, 12),
                     lambda_l1 = trial.suggest_float('lambda_l1', 0.01, 10.0, log = True),
                     lambda_l2 = trial.suggest_float('lambda_l2', 0.01, 10.0, log = True),
                     num_leaves = trial.suggest_int('num_leaves', 2, 100),
                     bagging_fraction = trial.suggest_float('bagging_fraction', 0.2, 0.9),
                     feature_fraction = trial.suggest_float('feature_fraction', 0.2, 0.9)
#                      device = 'gpu'
                    )

        scores = []
        
        skf = KFold(n_splits = 5, shuffle = True, random_state = self.seed)

        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = LGBMRegressor(**param).fit(X_train, Y_train)

            preds_valid = model.predict(X_valid)

            score = mean_squared_error(Y_valid, preds_valid, squared = False)
            scores.append(score)

        return np.mean(scores)
    
## Defining SEED and Trials
SEED = 42
N_TRIALS = 50

# Execute an optimization
study = optuna.create_study(direction = 'minimize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)

print('----------------------------')
print(' (-: Starting CV process :-)')
print('----------------------------')

lgb_cv_scores, preds = list(), list()

for i in range(5):

    skf = KFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building XGBoost model
        lgb_md = LGBMRegressor(**study.best_trial.params).fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict(X_test)
        lgb_pred_2 = lgb_md.predict(test_lgb)
        
        ## Computing rmse
        lgb_cv_scores.append(mean_squared_error(Y_test, lgb_pred_1, squared = False))
        preds.append(lgb_pred_2)

lgb_cv_score = np.mean(lgb_cv_scores)    
print('The average oof rmse score over 5-folds (run 5 times) is:', lgb_cv_score)

###############################
## Consolidating Predictions ##
###############################

lgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)
clean_pred = pd.DataFrame({'id': test_clean['id']})
clean_pred['price_clean'] = lgb_preds_test

submission.drop(columns = 'price', axis = 1, inplace = True)
submission = pd.merge(submission, clean_pred, on = 'id', how = 'left')
submission = pd.merge(submission, test_dup, on = 'id', how = 'left')

submission['price'] = np.where(np.isnan(submission['price_dup']), submission['price_clean'], submission['price_dup'])
submission.drop(columns = ['price_clean', 'price_dup'], axis = 1, inplace = True)

submission.to_csv('lgb_leakage_submission_1.csv', index = False)

print('--------------------------')    
print('...The process finished...')    
print('--------------------------')