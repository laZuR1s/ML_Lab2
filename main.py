import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

# 1. Чтение данных
data = pd.read_csv("student-mat.csv")

# 2. Целевая переменная (столбец для прогнозирования)
y = data['G3']

# 3. Выбираем остальные признаки
X = data.drop(['G3'], axis=1)

# Разделяем признаки по типу
num_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Категориальные — те, у кого количество уникальных значений <= 8
cat_cols = [cname for cname in X.columns
            if X[cname].dtype == "object"]  # только нечисловые признаки


print(f"Числовые столбцы ({len(num_cols)}): {num_cols}")
print(f"Категориальные столбцы ({len(cat_cols)}): {cat_cols}")

# 5. Импьютер для числовых данных
numerical_transformer = SimpleImputer(strategy='mean')

# 6. Конвейер для категориальных столбцов
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 7. Препроцессор (объединяем обработчики)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# 8. Итоговый конвейер с RandomForestRegressor
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=1))
])

# 9. Кросс-валидация
scores = -1 * cross_val_score(rf_model, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')
print("Средняя абсолютная ошибка (RandomForest, кросс-валидация):")
print(round(scores.mean(), 3))

# 10. Разделяем на обучающую и тестовую выборку
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=1)

# 11. Конвейер с XGBRegressor
xgb_model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=1))
])

# 12. Обучение и предсказание
xgb_model.fit(X_train, y_train)
preds = xgb_model.predict(X_valid)

# 13. Средняя абсолютная ошибка
mae = mean_absolute_error(y_valid, preds)
print("Средняя абсолютная ошибка (XGBRegressor):", round(mae, 3))

# 14. Вывод о сравнении
print("\nВывод:")
print("RandomForest — ансамбль деревьев, обученных независимо, хорошо работает при слабых нелинейностях.")
print("XGBRegressor — использует градиентный бустинг (поочередное улучшение деревьев),")
print("обычно показывает более высокую точность за счёт учёта ошибок предыдущих моделей.")
