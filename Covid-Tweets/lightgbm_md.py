import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from lightgbm import LGBMClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# test_id = test['Id']
# test = test.drop(columns = ['Id', 'text', 'reply_to_screen_name', 'hashtags'], axis = 1)

## Defining input and target
X = train.drop(columns = ['text', 'reply_to_screen_name', 'hashtags', 'country'], axis = 1)
Y = train['country']
Y = np.where(Y == 'us', 0, 
             np.where(Y == 'uk', 1, 
                      np.where(Y == 'canada', 2, 
                               np.where(Y == 'australia', 3,
                                        np.where(Y == 'ireland', 4, 5)))))

## Defining the hyper-parameter grid
LightGBM_param_grid = {'n_estimators': [300],
                       'max_depth': [5],
                       'num_leaves': [20],
                       'min_data_in_leaf': [15],
                       'learning_rate': [0.01],
                       'feature_fraction': [0.8],
                       'lambda_l1': [0],
                       'lambda_l2': [0]
                      }

## Building the multi-classifier  
one_vs_all_LightGBM = OneVsRestClassifier(estimator = LGBMClassifier(*LightGBM_param_grid)).fit(X, Y)

## Predicting on the test
one_vs_all_LightGBM_pred = one_vs_all_RF.predict_proba(test)
one_vs_all_LightGBM_pred = np.argmax(one_vs_all_LightGBM_pred, axis = 1)

data_out = pd.DataFrame({'Id': test_id, 'Category': one_vs_all_LightGBM_pred})
data_out['Category'] = np.where(data_out['Category'] == 0, 'us',
                                np.where(data_out['Category'] == 1, 'uk',
                                         np.where(data_out['Category'] == 2, 'canada',
                                                  np.where(data_out['Category'] == 3, 'australia',
                                                           np.where(data_out['Category'] == 4, 'ireland', 'new_zealand')))))

data_out.to_csv('LightGBM_submission_md.csv', index = False)