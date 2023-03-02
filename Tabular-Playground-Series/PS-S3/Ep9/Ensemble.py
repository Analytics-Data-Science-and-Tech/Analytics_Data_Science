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

class OptimizedEnsemble(object):
    
    def __init__(self):
        self.coef_ = 0

    def _rmse_loss(self, coef, X, y):
        
        ens = coef[0]*X[:, 0] + coef[1]*X[:, 1] + coef[2]*X[:, 2]
        ll = mean_squared_error(y, ens, squared = False)
        return ll

    def fit(self, X, y):
        loss_partial = partial(self._rmse_loss, X = X, y = y)
        initial_coef = [1/3, 1/3, 1/3]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')

    def predict(self, X, coef):
        
        ens = coef[0]*X[:, 0] + coef[1]*X[:, 1] + coef[2]*X[:, 2]
        return ens

    def coefficients(self):
        return self.coef_['x']
    

train_no_dup = train.drop(columns = 'id', axis = 1)
train_no_dup = pd.DataFrame(train_no_dup.groupby(train_no_dup.columns.tolist()[0:8])['Strength'].median()).reset_index()

X = train_no_dup.drop(columns = ['Strength'], axis = 1)
Y = train_no_dup['Strength']

test_baseline = test.drop(columns = ['id'], axis = 1)

ens_cv_scores, preds = list(), list()

for i in tqdm(range(5)):

    skf = KFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        #############    
        ## XGBoost ##
        #############
        
        XGB_md = XGBRegressor(tree_method = 'hist',
                              colsample_bytree = 0.671460244215802, 
                              gamma = 2.5281806276307384, 
                              learning_rate = 0.002046162779305807, 
                              max_depth = 8, 
                              min_child_weight = 80, 
                              n_estimators = 2690, 
                              subsample = 0.44886485549735244).fit(X_train, Y_train)
        
        ##############
        ## LightGBM ##
        ##############
        
        lgb_md = LGBMRegressor(n_estimators = 5420,
                               max_depth = 3,
                               learning_rate = 0.0014779400349972686,
                               num_leaves = 61,
                               lambda_l1 = 7.384172796287736,
                               lambda_l2 = 0.10456555506292783,
                               bagging_fraction = 0.22841166601766863,
                               feature_fraction = 0.659898030).fit(X_train, Y_train)
        
        ##############
        ## CatBoost ##
        ##############
        
        cat_md = CatBoostRegressor(loss_function = 'RMSE',
                                   iterations = 4738,
                                   learning_rate = 0.003143666241424718,
                                   depth = 4,
                                   random_strength = 0.29823973415192867,
                                   bagging_temperature = 0.3408793603898661,
                                   border_count = 112,
                                   l2_leaf_reg = 17,
                                   verbose = False).fit(X_train, Y_train)
        
        ######################
        ## Optimal Ensemble ##
        ######################
        
        XGB_pred_1 = XGB_md.predict(X_test)
        lgb_pred_1 = lgb_md.predict(X_test)
        cat_pred_1 = cat_md.predict(X_test)
        models_pred_oof = np.transpose((XGB_pred_1, lgb_pred_1, cat_pred_1))

        opt_ens = OptimizedEnsemble()
        opt_ens.fit(models_pred_oof, Y_test)
        coef = opt_ens.coefficients()
        
        ens_pred = opt_ens.predict(models_pred_oof, coef)
        ens_cv_scores.append(mean_squared_error(Y_test, ens_pred, squared = False))
        
        XGB_pred_2 = XGB_md.predict(test_baseline)
        lgb_pred_2 = lgb_md.predict(test_baseline)
        cat_pred_2 = cat_md.predict(test_baseline)
        models_pred = np.transpose((XGB_pred_2, lgb_pred_2, cat_pred_2))
        
        ens_preds = opt_ens.predict(models_pred, coef)
        preds.append(ens_preds)
        
ens_cv_score = np.mean(ens_cv_scores)    
print('The average oof rmse score over 5-folds (run 5 times) of the ensemble model is:', ens_cv_score)

ens_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)

submission['Strength'] = ens_preds_test
submission.to_csv('Ensemble_Optuna_baseline_submission.csv', index = False)