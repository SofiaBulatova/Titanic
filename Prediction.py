import pandas as pd
from Config import config
import joblib
from pathlib import Path
import json

def test(config):
    output_dir = Path(config.paths.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_info_path = output_dir / 'best_model_info.json'
    with open(best_info_path, 'r') as f:
        best_info = json.load(f)
    best_model_name = best_info['best_model']
    print(f"Загружена информация о модели: {best_model_name} с точностью {best_info['best_accuracy']:.4f}")

    model_path = output_dir / f'{best_model_name}.joblib'
    final_model = joblib.load(model_path)
    print(f"Модель загружена: {model_path}")

    test_df = pd.read_csv(config.paths.test_path)
    X_test = test_df.drop(columns=config.id_col)
    test_ids = test_df[config.id_col]

    test_preds = final_model.predict(X_test)
    submission = pd.DataFrame({config.id_col: test_ids, config.target_col: test_preds})
    submission_path = output_dir / 'submission.csv'
    submission.to_csv(submission_path, index=False)
    print(f"Сабмишн сохранён: {submission_path}")