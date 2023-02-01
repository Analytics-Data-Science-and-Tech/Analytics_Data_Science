def objective(trial):
    param = {
        'objective' : 'multiclass',
        'metric' : 'multi_logloss',
        "verbosity": -100,
        "boosting_type": "gbdt",            
        'is_unbalance': True,
        'n_estimators': trial.suggest_int('n_estimators', 300, 10000),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'reg_lambda': trial.suggest_float('lambda_l2', 0.01, 10.0, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 2, 512),
    }
    K = 5
    scores = []

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target])):
        print(fold, end=' ')
        X_train, X_valid = train.iloc[train_idx][features], train.iloc[valid_idx][features]
        y_train , y_valid = train[target].iloc[train_idx] , train[target].iloc[valid_idx]

        model = LGBMClassifier(**param, random_state=42)
        model.fit(X_train, y_train)

        preds_valid = model.predict(X_valid)
        score = cohen_kappa_score(y_valid,  preds_valid, weights = "quadratic")
        scores.append(score)
    return np.mean(scores)