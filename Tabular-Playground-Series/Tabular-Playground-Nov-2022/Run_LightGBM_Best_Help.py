import pandas as pd
import numpy as np

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

def Run_LightGBM_Best(X, Y, test_new, submission):

    ## Defining list to store results
    lgb_results = list()
    test_preds_lgb_fold_1 = list() 
    test_preds_lgb_fold_2 = list()
    test_preds_lgb_fold_3 = list()
    test_preds_lgb_fold_4 = list()
    test_preds_lgb_fold_5 = list()

    fold = 1
    kfold = StratifiedKFold(n_splits = 5, shuffle = True)

    for train_ix, test_ix in kfold.split(X, Y):

        ## Splitting the data 
        X_train, X_test = X.iloc[train_ix], X.iloc[test_ix]
        Y_train, Y_test = Y.iloc[train_ix], Y.iloc[test_ix]

        ## Building model
        lgb_md = LGBMClassifier(n_estimators = 1000, 
                                learning_rate = 0.01,
                                num_leaves = 50,
                                max_depth = 17, 
                                lambda_l1 = 3, 
                                lambda_l2 = 1, 
                                bagging_fraction = 0.4, 
                                feature_fraction = 0.4).fit(X_train, Y_train)

        ## Predicting on test
        lgb_pred = lgb_md.predict_proba(X_test)[:, 1]
        score = log_loss(Y_test, lgb_pred)
        
        if (score < 0.52):
            
            lgb_results.append(score)

            if (fold == 1):
                test_preds_lgb_fold_1.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 2):
                test_preds_lgb_fold_2.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 3):
                test_preds_lgb_fold_3.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 4):
                test_preds_lgb_fold_4.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 5):
                test_preds_lgb_fold_5.append(lgb_md.predict_proba(test_new)[:, 1])
                
        else:
            
            lgb_results.append(0)
            
            if (fold == 1):
                test_preds_lgb_fold_1.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 2):
                test_preds_lgb_fold_2.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 3):
                test_preds_lgb_fold_3.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 4):
                test_preds_lgb_fold_4.append(lgb_md.predict_proba(test_new)[:, 1])

            if (fold == 5):
                test_preds_lgb_fold_5.append(lgb_md.predict_proba(test_new)[:, 1])
            

        fold += 1

    array_scores = np.array(lgb_results)    
    print('The average log-loss over 5-fold CV is', np.mean(array_scores[array_scores > 0]))

    ##########################################
    ## Weighted average of fold predictions ##
    ##########################################
    
    w_init = w_fun(lgb_results)
    w1 = w_init[0]
    w2 = w_init[1]
    w3 = w_init[2]
    w4 = w_init[3]
    w5 = w_init[4]
    w_tot = w1 + w2 + w3+ w4 + w5
    
    if (w_tot > 0):
        
        w1 = w1 / w_tot
        w2 = w2 / w_tot
        w3 = w3 / w_tot
        w4 = w4 / w_tot
        w5 = w5 / w_tot

        pred1 = w1*test_preds_lgb_fold_1[0]
        pred2 = w2*test_preds_lgb_fold_2[0]
        pred3 = w3*test_preds_lgb_fold_3[0]
        pred4 = w4*test_preds_lgb_fold_4[0]
        pred5 = w4*test_preds_lgb_fold_5[0]

        submission['pred'] = pred1 + pred2 + pred3 + pred4 + pred5

        return [np.mean(array_scores[array_scores > 0]), sum(array_scores > 0), submission]
    
    else:
        
        return []


def w_fun(scores):
    
    w1 = w_fun_help(scores[0])
    w2 = w_fun_help(scores[1])
    w3 = w_fun_help(scores[2])
    w4 = w_fun_help(scores[3])
    w5 = w_fun_help(scores[4])
    
    return [w1, w2, w3, w4, w5]


def w_fun_help(value): 
    if (value > 0):        
        return 1/value  
    else:
        return 0