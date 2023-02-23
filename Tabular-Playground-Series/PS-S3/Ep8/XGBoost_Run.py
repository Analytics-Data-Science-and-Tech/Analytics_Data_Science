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
    
    return df

def feature_engineering(df):
    
    df['volume'] = df['x'] * df['y'] * df['z']
    df['surface_area'] = 2 * (df['x'] * df['y'] + df['y'] * df['z'] + df['z'] * df['x'])
    df['aspect_ratio_xy'] = df['x'] / df['y']
#     df['aspect_ratio_yz'] = df['y'] / df['z']
    df['aspect_ratio_zx'] = df['z'] / df['x']
    df['diagonal_distance'] = np.sqrt(df['x'] ** 2 + df['y'] ** 2 + df['z'] ** 2)
    df['relative_height'] = (df['z'] - df['z'].min()) / (df['z'].max() - df['z'].min())
    df['relative_position'] = (df['x'] + df['y'] + df['z']) / (df['x'] + df['y'] + df['z']).sum()
    df['volume_ratio'] = df['x'] * df['y'] * df['z'] / (df['x'].mean() * df['y'].mean() * df['z'].mean())
    df['length_ratio'] = df['x'] / df['x'].mean()
    df['width_ratio'] = df['y'] / df['y'].mean()
    df['height_ratio'] = df['z'] / df['z'].mean()
    df['sphericity'] = 1.4641 * (6 * df['volume'])**(2/3) / df['surface_area']
    df['compactness'] = df['volume']**(1/3) / df['x']
    
    return df

train_FE = updating_labels(train).drop(columns = ['cut', 'color', 'clarity'], axis = 1)
test_FE = updating_labels(test).drop(columns = ['cut', 'color', 'clarity'], axis = 1)


# train_FE = feature_engineering(updating_labels(train).drop(columns = ['cut', 'color', 'clarity'], axis = 1))
# test_FE = feature_engineering(updating_labels(test).drop(columns = ['cut', 'color', 'clarity'], axis = 1))


#######################
## Feature Selection ##
#######################

print('-----------------------------')
print('Feature Selection Started')
print('-----------------------------')


X = train_FE.drop(columns = ['id', 'price'], axis = 1)
Y = train_FE['price']

# ## Running RFECV multiple times
# RFE_results = list()

# for i in tqdm(range(0, 10)):
    
#     auto_feature_selection = RFECV(estimator = XGBRegressor(tree_method = 'hist'), step = 1, min_features_to_select = 2, cv = 5, scoring = 'neg_root_mean_squared_error').fit(X, Y)
    
#     ## Extracting and storing features to be selected
#     RFE_results.append(auto_feature_selection.support_)

# ## Changing to data-frame
# RFE_results = pd.DataFrame(RFE_results)
# RFE_results.columns = X.columns

# ## Computing the percentage of time features are flagged as important
# RFE_results = 100*RFE_results.apply(np.sum, axis = 0) / RFE_results.shape[0]

# ## Identifying features with a percentage score > 80%
# features_to_select = RFE_results.index[RFE_results > 80].tolist()

# features_dict = {'Features': features_to_select}
# pd.DataFrame(features_dict).to_csv('Important_features_1.csv', index = False)

# print(features_to_select)

############
## Optuna ##
############

print('-----------------------------')
print('Optuna Optimization Started')
print('-----------------------------')

# X = train_FE[features_to_select]
# Y = train_FE['price']

# test_xgb = test_FE[features_to_select]

X = train_FE.drop(columns = ['id', 'price'], axis = 1)
Y = train_FE['price']

test_xgb = test_FE.drop(columns = 'id', axis = 1)

class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        ## Parameters to be evaluated
        param = dict(objective = 'reg:squarederror',
                     eval_metric = 'rmse',
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
        
        skf = KFold(n_splits = 5, random_state = self.seed, shuffle = True)

        for train_idx, valid_idx in skf.split(X, Y):

            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[valid_idx]

            model = XGBRegressor(**param).fit(X_train, Y_train)
            preds_valid = model.predict(X_valid)
            
            score = mean_squared_error(Y_valid, preds_valid, squared = False)
            scores.append(score)
            
        return np.mean(scores)
    
## Defining number of runs and seed
SEED = 42
N_TRIALS = 50

# Execute an optimization
study = optuna.create_study(direction = 'minimize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)

print('The best trial rmse is: ', study.best_trial.values)
print('The best hyper-parameter combination is: ', study.best_trial.params)

optuna_hyper_params = pd.DataFrame.from_dict([study.best_trial.params])
file_name = 'XGB_FS_Seed_' + str(SEED) + '_Optuna_Hyperparameters_2.csv'
optuna_hyper_params.to_csv(file_name, index = False)


print('-----------------------------')
print('Starting CV process')
print('-----------------------------')

XGB_cv_scores, preds = list(), list()

for i in range(5):

    skf = KFold(n_splits = 5, random_state = 42, shuffle = True)
    
    for train_ix, test_ix in skf.split(X, Y):
        
        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]
                
        ## Building XGBoost model
        XGB_md = XGBRegressor(**study.best_trial.params,
                              tree_method = 'hist').fit(X_train, Y_train)
        
        ## Predicting on X_test and test
        XGB_pred_1 = XGB_md.predict(X_test)
        XGB_pred_2 = XGB_md.predict(test_xgb)
        
        ## Computing rmse
        XGB_cv_scores.append(mean_squared_error(Y_test, XGB_pred_1, squared = False))
        preds.append(XGB_pred_2)

XGB_cv_score = np.mean(XGB_cv_scores)    
print('The average oof rmse score over 5-folds (run 5 times) is:', XGB_cv_score)

xgb_preds_test = pd.DataFrame(preds).apply(np.mean, axis = 0)
submission['price'] = xgb_preds_test

submission.to_csv('XGBoost_baseline_5_submission.csv', index = False)

print('-----------------------------')    
print('The process finished...')    
print('-----------------------------')
