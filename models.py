from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from Config import config
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from NN import NeuralNet

# Создаем модель
def get_model(model_name, params, config=None):
    if model_name == 'stacking':
        estimator_names = params['estimators']
        estimators = []
        for name in estimator_names:
            base_params = dict(config.models[name])
            base_model = get_model(name, base_params, config)
            estimators.append((name, base_model))
        final_name = params['final_estimator']
        final_params = dict(config.models[final_name])
        final_estimator = get_model(final_name, final_params, config)
        cv = params.get('cv', 5)
        return StackingClassifier(estimators=estimators, final_estimator=final_estimator, cv=cv)
    elif model_name == "lasso":
        return LogisticRegression(**params)
    elif model_name == "ridge":
        return LogisticRegression(**params)
    elif model_name == "elasticnet":
        return LogisticRegression(**params)
    elif model_name == "knn":
        return KNeighborsClassifier(**params)
    elif model_name == "decision_tree":
        return DecisionTreeClassifier(**params)
    elif model_name == "random_forest":
        return RandomForestClassifier(**params)
    elif model_name == "catboost":
        return CatBoostClassifier(**params)
    elif model_name == "lightgbm":
        return LGBMClassifier(**params)
    elif model_name == "xgboost":
        return XGBClassifier(**params)
    elif model_name == "neural_network":
        return NeuralNet(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")

# Создаем экземпляр модели
def get_models_dict(config):
    models = {}
    for name, params in config.models.items():
        p = dict(params)
        model = get_model(name, p, config=config)
        # Маштабируем признаки
        if name in ['lasso', 'ridge', 'elasticnet', 'knn']:
            model = make_pipeline(StandardScaler(), model)
        elif name == 'stacking':
            model = make_pipeline(StandardScaler(), model)
        models[name] = model
    return models
