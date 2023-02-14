class Objective:

    def __init__(self, seed):
        # Hold this implementation specific arguments as the fields of the class.
        self.seed = seed

    def __call__(self, trial):
        ## Parameters to be evaluated
        param = dict(objective = 'reg:absoluteerror',
                     metric = '',
                     tree_method = 'gbdt', 
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
#         skf = KFold(n_splits = 5, shuffle = True, random_state = self.seed)

        for train_idx, valid_idx in skf.split(X, Y):

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
SEED = 33
N_TRIALS = 50

# Execute an optimization
study = optuna.create_study(direction = 'maximize')
study.optimize(Objective(SEED), n_trials = N_TRIALS)

print('-----------------------------')
print('Saving Optuna Hyper-Parameters')
print('-----------------------------')


optuna_hyper_params = pd.DataFrame.from_dict([study.best_trial.params])
file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_Optuna_Hyperparameters_1.csv'
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
    skf = StratifiedKFold(n_splits = 5, shuffle = True, random_state = SEED)
#     skf = KFold(n_splits = 5, random_state = SEED, shuffle = True)

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


    file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_Run_' + str(i) + '_1.csv' 
    submission.to_csv(file_name, index = False)

file_name = 'XGB_Reg_4_features_Seed_' + str(SEED) + '_CV_Score_1.csv'
CV_scores.to_csv(file_name, index = False)