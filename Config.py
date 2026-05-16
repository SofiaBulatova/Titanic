from omegaconf import OmegaConf

config = {
    'general': {
        'seed': 42,
    },
    'paths': {
        'train_path': 'Data/train_processed.csv',
        'test_path': 'Data/test_processed.csv',
        'output': 'output'
    },
    'models': {
        'lasso': {'l1_ratio': 1.0, 'solver':'liblinear', 'C':1.0, 'max_iter':1000},
        'ridge': {'l1_ratio': 0.0, 'solver':'lbfgs', 'C':1.0, 'max_iter':1000},
        'elasticnet': {'l1_ratio': 0.5, 'solver':'saga', 'l1_ratio':0.5, 'C':1.0, 'max_iter':1000},
        'knn': {'n_neighbors': 5},
        'decision_tree': {'max_depth': 10, 'random_state': '${general.seed}'},
        'random_forest': {'n_estimators': 100, 'random_state': '${general.seed}'},
        'catboost': {'iterations': 100, 'verbose': False, 'random_state': '${general.seed}'},
        'lightgbm': {'n_estimators': 100, 'random_state': '${general.seed}'},
        'xgboost': {'n_estimators': 100, 'verbosity': 0, 'random_state': '${general.seed}'},
        'stacking': {'type': 'stacking', 'final_estimator': 'lasso', 'cv': 5, 'estimators': ['catboost', 'xgboost']},
        'neural_network': {
            'hidden_dims': [64, 32],
            'dropout': 0.3,
            'lr': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 15,
            'val_fraction': 0.15,
            'random_state': '${general.seed}',
            'step_size': 10,
            'gamma': 0.9
        },
    },
    'split': {
        'methods':['cv', 'holdout'],
        'cv':{'n_splits': 5, 'shuffle': True, 'random_state': '${general.seed}'},
        'holdout':{'test_size': 0.2, 'random_state': '${general.seed}'}

    },
    'id_col': 'PassengerId',
    'target_col': 'Survived'

}
config = OmegaConf.create(config)