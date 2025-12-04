import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# 1. Чтение данных
data = pd.read_csv("student-mat.csv")

# 2. Целевая переменная
y = data['G3']

# 3. Признаки
X = data.drop(['G3'], axis=1)

# Разделение по типам
num_cols = [c for c in X.columns if X[c].dtype in ['int64', 'float64']]
cat_cols = [c for c in X.columns if X[c].dtype == 'object']

print(f"Числовые столбцы ({len(num_cols)}): {num_cols}")
print(f"Категориальные столбцы ({len(cat_cols)}): {cat_cols}")

# 4. Наборы трансформеров

# Для числовых
num_transformers = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "scaled_mean": Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
}

# Для категориальных
cat_transformers = {
    "freq": Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]),
    "constant": Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
}

# 5. Перебор всех комбинаций для выбора лучшего preprocessora

best_score = float("inf")
best_name = None
best_preprocessor = None

for num_name, num_trans in num_transformers.items():
    for cat_name, cat_trans in cat_transformers.items():

        name = f"num:{num_name} + cat:{cat_name}"

        preprocessor = ColumnTransformer([
            ("num", num_trans, num_cols),
            ("cat", cat_trans, cat_cols)
        ])

        model = Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestRegressor(random_state=1))
        ])

        score = -cross_val_score(
            model, X, y,
            scoring='neg_mean_absolute_error',
            cv=5
        ).mean()

        print(f"{name} → MAE = {score:.3f}")

        if score < best_score:
            best_score = score
            best_name = name
            best_preprocessor = preprocessor

print("\nЛУЧШАЯ КОМБИНАЦИЯ ПРЕПРОЦЕССОРОВ:")
print(best_name, " → MAE =", round(best_score, 3))

# 7. Препроцессор (объединяем обработчики)
preprocessor = best_preprocessor

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
print("XGBRegressor — использует градиентный бустинг, улучшая деревья последовательно.")
print("Использование лучшего препроцессора — повышает точность обеих моделей.")
