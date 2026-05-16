import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
from Config import config

# Выбор метода и формирование Датафрейма
def evaluate_all_models(models_dict, X, y, config):
    rows = []
    for method in config.split.methods:
        if method == 'cv':
            cv_config = config.split.cv
            for name, model in models_dict.items():
                acc = get_kfold(model, X, y, n_folds=cv_config.n_splits,
                                shuffle=cv_config.shuffle,
                                random_state=cv_config.random_state)
                rows.append({'model': name, 'method': method, 'accuracy': acc})
        elif method == 'holdout':
            holdout_config = config.split.holdout
            for name, model in models_dict.items():
                acc = accuracy_holdout(model, X, y, test_size=holdout_config.test_size,
                                       random_state=holdout_config.random_state)
                rows.append({'model': name, 'method': method, 'accuracy': acc})
    df = pd.DataFrame(rows)
    df = df.sort_values('accuracy', ascending=False).reset_index(drop=True)
    return df

# Осуществляем StratifiedKFold
def get_kfold(model, X, y, n_folds, shuffle, random_state):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')
    return scores.mean()

# Осуществляем train_test_split
def accuracy_holdout(model, X, y, test_size, random_state):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)