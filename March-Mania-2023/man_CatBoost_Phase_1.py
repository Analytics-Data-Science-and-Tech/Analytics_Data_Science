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
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier, LGBMRegressor 
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import optuna 

s3 = boto3.resource('s3')
bucket_name = 'analytics-data-science-competitions'
bucket = s3.Bucket(bucket_name)

file_key_1 = 'March-Mania-2023/MNCAA_train.csv'
file_key_2 = 'March-Mania-2023/MNCAA_test.csv'
file_key_3 = 'March-Mania-2023/WNCAA_train.csv'
file_key_4 = 'March-Mania-2023/WNCAA_test.csv'

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

## Reading data files
man_train = pd.read_csv(file_content_stream_1)
man_test = pd.read_csv(file_content_stream_2)
woman_train = pd.read_csv(file_content_stream_3)
woman_test = pd.read_csv(file_content_stream_4)

man_train['target'] = np.where(man_train['ResultDiff'] > 0, 1, 0)
woman_train['target'] = np.where(woman_train['ResultDiff'] > 0, 1, 0)

############
## Optuna ##
############

print('------------------------------------')
print(' (-: Optuna Optimization Started :-)')
print('------------------------------------')

class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        
        ## Parameters to be evaluated
        param = dict(iterations = trial.suggest_int('iterations', 300, 10000),
                     learning_rate = trial.suggest_float('learning_rate', 0.001, 1, log = True),
                     depth = trial.suggest_int('depth', 3, 12),
                     random_strength = trial.suggest_float('random_strength', 0.01, 10.0, log = True),
                     bagging_temperature = trial.suggest_float('bagging_temperature', 0.01,  0.99),
                     border_count = trial.suggest_int('border_count', 1, 255),
                     l2_leaf_reg = trial.suggest_int('l2_leaf_reg', 2, 30),
                     verbose = False
                    )

        scores = []
        
        for i in range(2010, 2022):
    
            train_data = man_train[man_train['Season'] <= i].reset_index(drop = True) 
    
            if ((i + 1) == 2020): 
                continue 
            else:
                test_data = man_train[man_train['Season'] == (i + 1)].reset_index(drop = True)
    
            X_train = train_data.drop(columns = ['Season', 'T1', 'T2', 'T1_Points', 'T2_Points', 'ResultDiff', 'target'], axis = 1)
            Y_train = train_data['ResultDiff']
            X_valid = test_data.drop(columns = ['Season', 'T1', 'T2', 'T1_Points', 'T2_Points', 'ResultDiff', 'target'], axis = 1)
            Y_valid = test_data['ResultDiff']
        
            model = CatBoostRegressor(**param).fit(X_train, Y_train)
            preds_valid = model.predict(X_valid)

            score = mean_squared_error(Y_valid, preds_valid)
            scores.append(score)

        return np.mean(scores)
    
## Defining SEED and Trials
SEED = 42
N_TRIALS = 50

# Execute an optimization
study = optuna.create_study(direction = 'minimize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)

print('----------------------------------------')
print(' (-: Saving Optuna Hyper-Parameters :-) ')
print('----------------------------------------')

optuna_hyper_params = pd.DataFrame.from_dict([study.best_trial.params])
file_name = 'man_CatBoost_Phase_1_' + str(SEED) + '_Optuna_Hyperparameters.csv'
optuna_hyper_params.to_csv(file_name, index = False)


# ## Building model with optuna parameters
# X = man_train.drop(columns = ['Season', 'T1', 'T2', 'T1_Points', 'T2_Points', 'ResultDiff', 'target'], axis = 1)
# Y = man_train['ResultDiff']

# xgb_md = XGBRegressor(**study.best_trial.params).fit(X, Y)

# xgb_pred_test = xgb_md.predict(man_test.drop(columns = ['ID''Season', 'T1', 'T2'], axis = 1))
# man_test['ResultDiff'] = round(xgb_pred_test)
# man_test.to_csv('man_test_xgb.csv', index = False)
