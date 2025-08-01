import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def custom_rfecv_catboost_timeseries(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: List[str],
    model_params: Dict[str, Any],
    cv_splitter: TimeSeriesSplit,
    min_features_to_select: int = 1,
    step: int = 1,
    scoring: str = 'neg_root_mean_squared_error'
) -> List[str]:
    """
    Выполняет рекурсивный отбор признаков для CatBoost с кросс-валидацией
    для временных рядов (TimeSeriesSplit) и поддержкой NaN.

    Args:
        X (pd.DataFrame): Датафрейм с признаками, отсортированный по времени.
        y (pd.Series): Целевая переменная, отсортированная по времени.
        cat_features (List[str]): Список названий категориальных колонок.
        model_params (Dict[str, Any]): Параметры для CatBoostClassifier/Regressor.
        cv_splitter (TimeSeriesSplit): Инициализированный объект TimeSeriesSplit.
        min_features_to_select (int): Минимальное количество признаков.
        step (int): Количество признаков для удаления на каждой итрации.
        scoring (str): Метрика для оценки качества.

    Returns:
        List[str]: Список названий лучших отобранных признаков.
    """
    print("Запущен RFECV с TimeSeriesSplit и поддержкой NaN...")
    
    features_to_select = X.columns.tolist()
    history = {} # {количество_фичей: средний_скор}

    while len(features_to_select) >= min_features_to_select:
        num_features = len(features_to_select)
        print(f"Текущее количество признаков: {num_features}")

        # 1. Оценка качества на кросс-валидации
        model = CatBoostClassifier(**model_params, cat_features=cat_features)
        
        # Используем переданный cv_splitter
        scores = cross_val_score(
            estimator=model,
            X=X[features_to_select],
            y=y,
            cv=cv_splitter,
            scoring=scoring
        )
        mean_score = np.mean(scores)
        history[num_features] = mean_score
        print(f"  Средняя метрика ({scoring}): {mean_score:.4f}")

        if len(features_to_select) == min_features_to_select:
            break

        # 2. Обучение на всех данных для получения важности
        model.fit(X[features_to_select], y)
        importances = pd.Series(model.get_feature_importance(), index=features_to_select)
        
        # 3. Удаление наименее важных признаков
        least_important = importances.nsmallest(step).index.tolist()
        features_to_select = [f for f in features_to_select if f not in least_important]
        
    print("\nПроцесс отбора завершен.")

    # 4. Выбор лучшего набора признаков
    # Для метрик ошибок (типа MSE) нужно искать минимум, для остальных (AUC, Accuracy) - максимум.
    if "neg" in scoring or "error" in scoring or "loss" in scoring:
        best_num_features = min(history, key=history.get)
    else:
        best_num_features = max(history, key=history.get)
        
    best_score = history[best_num_features]
    print(f"\nОптимальное количество признаков: {best_num_features} (с метрикой {best_score:.4f})")
    
    # Восстанавливаем лучший набор признаков
    final_features = X.columns.tolist()
    while len(final_features) > best_num_features:
        model_final = CatBoostClassifier(**model_params, cat_features=cat_features)
        model_final.fit(X[final_features], y)
        importances = pd.Series(model_final.get_feature_importance(), index=final_features)
        least_important_feature = importances.nsmallest(1).index.tolist()[0]
        final_features.remove(least_important_feature)

    return final_features

if __name__ == '__main__':
    from sklearn.datasets import make_classification

    # 1. Генерируем данные
    X_raw, y_raw = make_classification(
        n_samples=500, n_features=25, n_informative=6, n_redundant=8,
        n_classes=2, flip_y=0.05, random_state=42
    )
    # Важно: Для TimeSeriesSplit данные должны быть упорядочены по времени.
    # Мы симулируем это, используя стандартный индекс.
    X = pd.DataFrame(X_raw, columns=[f'f_{i}' for i in range(25)]).sort_index()
    y = pd.Series(y_raw, name='target').sort_index()

    # 2. Создаем категориальные признаки
    cat_cols = ['f_20', 'f_21']
    for col in cat_cols:
        X[col] = pd.cut(X[col], bins=4, labels=[f'c{i}' for i in range(4)]).astype(str)
        
    # 3. Вносим пропуски в данные
    for col in ['f_5', 'f_15', 'f_20']:
        nan_mask = X.sample(frac=0.15, random_state=ord(col[2])).index
        X.loc[nan_mask, col] = np.nan
    print(f"Количество пропусков:\n{X.isnull().sum()[X.isnull().sum() > 0].to_string()}")

    # 4. ⚙️ Инициализируем TimeSeriesSplit
    # n_splits=5 означает, что будет 5 фолдов.
    # На первом фолде обучение на 1/6 данных, тест на 2/6.
    # На втором фолде обучение на 2/6 данных, тест на 3/6 и т.д.
    tscv_splitter = TimeSeriesSplit(n_splits=5)

    # 5. Определяем параметры модели
    catboost_params = {
        'iterations': 100,
        'verbose': 0,
        'random_seed': 42
    }
    
    # 6. Запускаем отбор признаков
    best_features = custom_rfecv_catboost_timeseries(
        X=X,
        y=y,
        cat_features=cat_cols,
        model_params=catboost_params,
        cv_splitter=tscv_splitter, # Передаем наш сплиттер
        min_features_to_select=3,
        step=1,
        scoring='roc_auc' # Для классификации
    )

    # 7. Выводим результат
    print("\n--- Итоговый результат ---")
    print("Лучшие отобранные признаки:", best_features)
    print(f"\nРазмерность исходных данных: {X.shape}")
    print(f"Размерность данных после отбора: {(X.shape[0], len(best_features))}")
