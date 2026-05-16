import pandas as pd
from Config import config
import joblib
from pathlib import Path
from models import get_models_dict
from Split import evaluate_all_models
import json
from omegaconf import OmegaConf

def train(config):
    # Cоздаем директорию
    output_dir = Path(config.paths.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Загрузка данных
    train_df = pd.read_csv(config.paths.train_path)
    X_train = train_df.drop(columns=[config.target_col])
    y_train = train_df[config.target_col]

    # Получаем словарь моделей
    models = get_models_dict(config)

    # Оценка всех моделей по всем методам из конфига
    results_df = evaluate_all_models(models, X_train, y_train, config)
    print("\n=== Результаты валидации ===")
    print(results_df.to_string(index=False))

    # Сохраняем результаты
    results_df.to_csv(output_dir / 'validation_results.csv', index=False)

    # Выбираем лучшую модель
    best_model = results_df.iloc[0]["model"]
    best_method = results_df.iloc[0]["method"]
    best_accuracy = float(results_df.iloc[0]["accuracy"])
    print(f"\nЛучшая модель по методу '{best_method}': {best_model} (accuracy = {best_accuracy:.4f})")

    # Обучаем лучшую модель на всех данных
    final_model = models[best_model]
    final_model.fit(X_train, y_train)

    # Сохраняем модель и параметры
    model_path = output_dir / f'{best_model}.joblib'
    joblib.dump(final_model, model_path)
    print(f"Модель сохранена: {model_path}")

    # Сохраняем параметры лучшей модели в JSON
    best_params = config.models[best_model]
    best_params_plain = OmegaConf.to_container(best_params, resolve=True)
    # Добавляем информацию о выбранной модели и метрике
    best_info = {
        'best_model': best_model,
        'best_accuracy': (best_accuracy),
        'validation_method': best_method,
        'params': best_params_plain
    }

    with open(output_dir / 'best_model_info.json', 'w') as f:
        json.dump(best_info, f, indent=4)