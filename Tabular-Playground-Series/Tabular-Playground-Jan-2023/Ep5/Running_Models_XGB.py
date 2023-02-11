import boto3
import pandas as pd; pd.set_option('display.max_columns', 100)
import numpy as np
import scipy as sp
from tqdm import tqdm
from functools import partial
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBRegressor

import optuna 

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'Tabular-Playground-Series/TS-S3-Ep5/train.csv'
file_key_2 = 'Tabular-Playground-Series/TS-S3-Ep5/test.csv'
file_key_3 = 'Tabular-Playground-Series/TS-S3-Ep5/sample_submission.csv'

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

## Enginering features
train['alcohol_density'] = train['alcohol'] * train['density']
train['sulphate/density'] = train['sulphates']  / train['density']
train['alcohol_sulphate'] = train['alcohol'] * train['sulphates']

test['alcohol_density'] = test['alcohol']  * test['density']
test['sulphate/density'] = test['sulphates']  / test['density']
test['alcohol_sulphate'] = test['alcohol'] * test['sulphates']

test_md = test.copy()

X = train[['sulphate/density', 'alcohol_density', 'alcohol', 'sulphates']]
Y = train['quality'] 

test_md = test_md[['sulphate/density', 'alcohol_density', 'alcohol', 'sulphates']]


class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = 0

    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 3
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 4
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 5
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 6
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 7
            else:
                X_p[i] = 8

        ll = cohen_kappa_score(y, X_p, weights = 'quadratic')
        return -ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X = X, y = y)
        initial_coef = [3.5, 4.5, 5.5, 6.5, 7.5]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')

    def predict(self, X, coef):
        X_p = np.copy(X)
        for i, pred in enumerate(X_p):
            if pred < coef[0]:
                X_p[i] = 3
            elif pred >= coef[0] and pred < coef[1]:
                X_p[i] = 4
            elif pred >= coef[1] and pred < coef[2]:
                X_p[i] = 5
            elif pred >= coef[2] and pred < coef[3]:
                X_p[i] = 6
            elif pred >= coef[3] and pred < coef[4]:
                X_p[i] = 7
            else:
                X_p[i] = 8
        return X_p

    def coefficients(self):
        return self.coef_['x']
    

class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        ## Parameters to be evaluated
        param = dict(objective = 'reg:absoluteerror',
                     eval_metric = 'mae',
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

        skf = KFold(n_splits = 5, shuffle = True, random_state = self.seed)

        for fold, (train_idx, valid_idx) in enumerate(skf.split(X, Y)):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train , Y_valid = Y.iloc[train_idx] , Y.iloc[valid_idx]

            model = XGBRegressor(**param).fit(X_train, Y_train)

            preds_train = model.predict(X_train)
            preds_valid = model.predict(X_valid)

            optR = OptimizedRounder()
            optR.fit(preds_train, Y_train)
            coef = optR.coefficients()
            preds_valid = optR.predict(preds_valid, coef).astype(int)

            score = cohen_kappa_score(Y_valid,  preds_valid, weights = 'quadratic')
            scores.append(score)

        return np.mean(scores)
    
## Defining number of runs and seed
RUNS = 50
SEED = 1
N_TRIALS = 50

# Execute an optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)

print('-----------------------------')
print('Saving Optuna Hyper-Parameters')
print('-----------------------------')


optuna_hyper_params = pd.DataFrame.from_dict([study.best_trial.params])
file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_Optuna_Hyperparameters_2.csv'
optuna_hyper_params.to_csv(file_name, index = False)

print('-----------------------------')
print('Starting CV process')
print('-----------------------------')


XGB_cv_score = list()
preds = list()

CV_scores = pd.DataFrame({'Run': [i for i in range(1, (RUNS + 1))]})
CV_scores['CV_Score'] = np.nan

for i in tqdm(range(RUNS)):

    XGB_cv_scores = list()
    skf = KFold(n_splits = 5, random_state = SEED, shuffle = True)

    for train_ix, test_ix in skf.split(X, Y):

        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]

        ## Building RF model
        XGB_md = XGBRegressor(**study.best_trial.params, 
                              random_state = i).fit(X_train, Y_train)

        ## Predicting on X_test and test
        XGB_pred_train = XGB_md.predict(X_train)
        XGB_pred_1 = XGB_md.predict(X_test)
        XGB_pred_2 = XGB_md.predict(test_md)
        
        ## Applying Optimal Rounder (using abhishek approach)
        optR = OptimizedRounder()
#         optR.fit(XGB_pred_1, Y_test)
        optR.fit(XGB_pred_train, Y_train)
        coef = optR.coefficients()
        XGB_pred_1 = optR.predict(XGB_pred_1, coef).astype(int)
        XGB_pred_2 = optR.predict(XGB_pred_2, coef).astype(int)
        
        ## Computing weighted quadratic kappa
        XGB_cv_scores.append(cohen_kappa_score(Y_test, XGB_pred_1, weights = 'quadratic'))
        preds.append(XGB_pred_2)
    
    avg_score = np.mean(XGB_cv_scores)
    print('The average oof weighted kappa score over 5-folds is:', avg_score)
    CV_scores.loc[i, 'CV_Score'] = avg_score
    
    XGB_preds_test = pd.DataFrame(preds).mode(axis = 0).loc[0, ]
    submission['quality'] = XGB_preds_test.astype(int)


    file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_Run_' + str(i) + '_2.csv' 
    submission.to_csv(file_name, index = False)

file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_Run_' + str(i) + '_CV_Score_2.csv'
CV_scores.to_csv(file_name, index = False)
    
print('-----------------------------')    
print('The process finished...')    
print('-----------------------------')
# ## Identifying the best random_state
# rand_state = np.argmax(XGB_cv_score)

# ## Building model in entire train dataset
# XGB_md = XGBRegressor(**study.best_trial.params, 
#                       random_state = rand_state).fit(X, Y)

# optR = OptimizedRounder()
# optR.fit(XGB_md.predict(X), Y)
# coef = optR.coefficients()
# XGB_pred = XGB_md.predict(test_md)
# XGB_pred = optR.predict(XGB_pred, coef).astype(int)

# submission['quality'] = XGB_pred

# file_name = 'XGB_Reg_FUll_Seed_' + str(SEED) + '.csv' 
# submission.to_csv(file_name, index = False)