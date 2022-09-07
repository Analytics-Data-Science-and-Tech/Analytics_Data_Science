import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_id = test['Id']
test = test.drop(columns = ['Id', 'text', 'reply_to_screen_name', 'hashtags'], axis = 1)

## Defining input and target
X = train.drop(columns = ['text', 'reply_to_screen_name', 'hashtags', 'country'], axis = 1)
Y = train['country']
Y = np.where(Y == 'us', 0, 
             np.where(Y == 'uk', 1, 
                      np.where(Y == 'canada', 2, 
                               np.where(Y == 'australia', 3,
                                        np.where(Y == 'ireland', 4, 5)))))

## Defining hyper-paramerters for RF
RF_param_grid = {'estimator__n_estimators': [500],
                 'estimator__min_samples_split': [10, 15],
                 'estimator__min_samples_leaf': [5, 7],
                 'estimator__max_depth' : [3, 5, 7]}


## Building the multi-classifier  
RF_md = OneVsRestClassifier(estimator = RandomForestClassifier())
one_vs_all_RF = GridSearchCV(RF_md, RF_param_grid, scoring = 'accuracy', n_jobs = -1, verbose = 3).fit(X, Y)

## Predicting on the test
one_vs_all_RF_pred = one_vs_all_RF.predict_proba(test)
one_vs_all_RF_pred = np.argmax(one_vs_all_RF_pred, axis = 1)

data_out = pd.DataFrame({'Id': test_id, 'Category': one_vs_all_RF_pred})
data_out['Category'] = np.where(data_out['Category'] == 0, 'us',
                                np.where(data_out['Category'] == 1, 'uk',
                                         np.where(data_out['Category'] == 2, 'canada',
                                                  np.where(data_out['Category'] == 3, 'australia',
                                                           np.where(data_out['Category'] == 4, 'ireland', 'new_zealand')))))

data_out.to_csv('RF_submission_3.csv', index = False)