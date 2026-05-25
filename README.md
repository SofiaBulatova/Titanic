# Titanic
ML-пайплайн для задачи Kaggle Titanic.
## Структура проекта
* Data/train_processed.csv - train после EDA.
* Data/test_processed.csv - test после EDA.
* output/best_model_info.json - информация о лучшей модели, включая ее гиперпараметры.
* output/neural_network.joblib - лучшая модель с сохраненными параметрами и гиперпараметрами.
* output/validation_resalts.csv - accuracy всех моделей.
* output/submission.csv - файл для загрузки на Kaggle.
* Config.py - ключевые настройки.
* Data analysis.py - анализ данных.
* Feature engineering.py - очистка данных, выбросы.
* NN.py - нейронная сеть.
* Prediction.py - предсказание лучшей модели.
* Split.py - K-Fold, train_test_split.
* main.py - основной файл для запуска.
* models.py - все модели
* train.py - обучение моделей, выбор лучшей и ее сохранение.
## Использованные модели
В пайплайне использовались следующие модели:
* Логистическая регрессия с Lasso.
* Логистическая регрессия с Ridge.
* Логистическая регрессия с ElasticNet.
* KNN.
* Дерево решений.
* Случайный лес.
* CatBoost.
* lightgbm.
* xgboost.
* Stacking c CatBoost и lightgbm
* Нейронная сеть.
## Результаты
По результатам валидации была выбрана Нейронная сеть с методом K-Fold и accuracy 0,8328.  
На Kaggle Public Score - 0,78229
