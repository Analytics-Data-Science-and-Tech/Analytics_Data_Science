import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np

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
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna as op

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

train_lgb = train.copy()
test_lgb = test.copy()

X = train_lgb.drop(columns = ['id', 'booking_status'], axis = 1)
Y = train_lgb['booking_status']

test_lgb = test_lgb.drop(columns = ['id'], axis = 1)


class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        ## Parameters to be evaluated
        param = dict(objective = 'binary',
                     metric = 'auc',
                     tree_method = 'gbdt', 
                     n_estimators = trial.suggest_int('n_estimators', 300, 10000),
                     learning_rate = trial.suggest_float('learning_rate', 0.001, 1, log = True),
                     max_depth = trial.suggest_int('max_depth', 3, 12),
                     lambda_l1 = trial.suggest_float('lambda_l1', 0.01, 10.0, log = True),
                     lambda_l2 = trial.suggest_float('lambda_l2', 0.01, 10.0, log = True),
                     num_leaves = trial.suggest_int('num_leaves', 2, 100),
                     bagging_fraction = trial.suggest_float('bagging_fraction', 0.2, 0.9),
                     feature_fraction = trial.suggest_float('feature_fraction', 0.2, 0.9)
                    )

        scores = []
        
        skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = self.seed)
#         skf = KFold(n_splits = 5, shuffle = True, random_state = self.seed)

        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = LGBMClassifier(**param).fit(X_train, Y_train)

#             preds_train = model.predict(X_train)
            preds_valid = model.predict_proba(X_valid)[:, 1]

            score = roc_auc_score(Y_valid, preds_valid)
            scores.append(score)

        return np.mean(scores)
    
## Defining number of runs and seed
RUNS = 2
SEED = 42
N_TRIALS = 10

# Execute an optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)

print('-----------------------------')
print('Saving Optuna Hyper-Parameters')
print('-----------------------------')

# optuna_hyper_params = pd.DataFrame.from_dict([study.best_trial.params])
# file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_Optuna_Hyperparameters_1.csv'
# optuna_hyper_params.to_csv(file_name, index = False)

print('-----------------------------')
print('Starting CV process')
print('-----------------------------')


CV_scores = pd.DataFrame({'Run': [i for i in range(1, (RUNS + 1))]})
CV_scores['CV_Score'] = np.nan

cv_scores, roc_auc_scores = list(), list()
preds = list()

for i in tqdm(range(RUNS)):

    XGB_cv_scores = list()
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)
#     skf = KFold(n_splits = 5, random_state = SEED, shuffle = True)

    for train_ix, test_ix in skf.split(X, Y):

        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]

        ## Building RF model
        lgb_md = LBGMClassifier(**study.best_trial.params, 
                                random_state = i).fit(X_train, Y_train)

        ## Predicting on X_test and test
        lgb_pred_1 = lgb_md.predict(X_test)
        lgb_pred_2 = lgb_md.predict(test_md)
        
        ## Computing roc-auc score
        roc_auc_scores.append(roc_auc_score(Y_test, lgb_pred_1))
        preds.append(lgb_pred_2)
    
    cv_scores.append(np.mean(roc_auc_scores))

    lgb_cv_score = np.mean(cv_scores)    
print('The roc-auc score over 5-folds (run 5 times) is:', lgb_cv_score)

    
    XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ]
    submission['quality'] = XGB_preds_test.astype(int)


    file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_Run_' + str(i) + '_1.csv' 
    submission.to_csv(file_name, index = False)

file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_CV_Score_1.csv'
CV_scores.to_csv(file_name, index = False)