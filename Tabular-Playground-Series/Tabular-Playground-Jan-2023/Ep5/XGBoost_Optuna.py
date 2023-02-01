def objective(trial):
    param = dict(
    objective = 'multi:softmax',
    eval_metric = 'mlogloss',
    tree_method='gpu_hist', 
    random_state=42,
    max_depth=trial.suggest_int("max_depth", 2, 10),
    learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
    n_estimators=trial.suggest_int("n_estimators", 30, 10000),
    subsample=trial.suggest_float("subsample", 0.2, 1.0),
    reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e1, log=True),
)
    K = 3
    scores = []

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=42)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(train[features], train[target])):
        print(fold, end=' ')
        X_train, X_valid = train.iloc[train_idx][features], train.iloc[valid_idx][features]
        y_train , y_valid = train[target].iloc[train_idx] , train[target].iloc[valid_idx]

        model = XGBClassifier(**param)
        model.fit(X_train, y_train)

        preds_valid = model.predict(X_valid)
        score = cohen_kappa_score(y_valid,  preds_valid, weights = "quadratic")
        scores.append(score)
    return np.mean(scores)