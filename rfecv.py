import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
import optuna
from copy import deepcopy

# --- Пример данных (как и раньше, с пропусками) ---
data = {
    'feature1': [1, 2, None, 4, 5],
    'feature2': [10, 20, 30, None, 50],
    'feature3': ['A', 'B', 'A', 'C', None],
    'feature4': [100, 200, 300, 400, 500],
    'feature5': [1, 0, 1, 0, 1],
    'feature6': [5, 4, 3, 2, 1],
    'feature7': [None, 7, 8, 9, 10],
    'target': [0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df.drop('target', axis=1)
y = df['target']

categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

# --- Вспомогательная функция для оценки модели с кросс-валидацией ---
def evaluate_model_with_cv(X_data, y_data, cat_features_list, model_params, cv_strategy, current_trial=None):
    """
    Оценивает модель с CatBoost с помощью кросс-валидации,
    возвращая среднюю LogLoss.
    """
    oof_preds = []
    oof_y = []

    for fold, (train_idx, val_idx) in enumerate(cv_strategy.split(X_data, y_data)):
        X_train_fold, X_val_fold = X_data.iloc[train_idx].copy(), X_data.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y_data.iloc[train_idx].copy(), y_data.iloc[val_idx].copy()

        # Если X_train_fold или X_val_fold пусты, это может случиться на последних итерациях отбора,
        # если все признаки были удалены, или из-за неудачного разбиения CV.
        # В таком случае, метрика не может быть посчитана.
        if X_train_fold.empty or X_val_fold.empty:
            return float('inf') # Возвращаем очень плохое значение

        # Создаем Pool для CatBoost, указывая категориальные признаки
        train_pool = Pool(X_train_fold, y_train_fold, cat_features=[f for f in cat_features_list if f in X_train_fold.columns])
        val_pool = Pool(X_val_fold, y_val_fold, cat_features=[f for f in cat_features_list if f in X_val_fold.columns])
        
        # Создаем новую модель для каждого фолда
        model = CatBoostClassifier(**model_params)
        model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=model_params.get('early_stopping_rounds', None), verbose=0)
        
        oof_preds.extend(model.predict_proba(X_val_fold)[:, 1])
        oof_y.extend(y_val_fold)
        
        if current_trial:
            # Отчет о промежуточной метрике для Optuna (если нужно)
            # Внимание: report() должен получать текущую метрику, а не итоговую.
            # Если метрика агрегируется по всем фолдам, то report нужно вызывать после агрегации.
            # Для pruning на основе фолдов, это корректно.
            current_trial.report(log_loss(y_val_fold, model.predict_proba(X_val_fold)[:, 1]), fold)
            if current_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return log_loss(oof_y, oof_preds)


# --- Обновленная функция рекурсивного отбора признаков по метрике ---
def recursive_feature_selection_by_metric(
    X_original,
    y_original,
    categorical_features,
    model_params,
    cv_strategy,
    max_features_to_remove=None, # Максимальное количество признаков для удаления
    patience_rounds=3 # Сколько раундов ухудшения метрики терпеть до остановки
):
    """
    Выполняет рекурсивный отбор признаков на основе метрики,
    автоматически подбирая оптимальное количество признаков.

    Args:
        X_original (pd.DataFrame): Исходные признаки (могут содержать пропуски).
        y_original (pd.Series): Целевая переменная.
        categorical_features (list): Список названий категориальных признаков.
        model_params (dict): Параметры CatBoostClassifier.
        cv_strategy (sklearn.model_selection._split._BaseKFold): Стратегия кросс-валидации.
        max_features_to_remove (int, optional): Максимальное количество признаков, которое можно удалить.
                                                Если None, удаляет до 1 признака.
        patience_rounds (int): Количество раундов без улучшения, после которых отбор останавливается.

    Returns:
        tuple: (list_of_best_features, best_metric_score)
    """
    
    initial_features = list(X_original.columns)
    if not initial_features:
        return [], float('inf')

    # Оценим начальную метрику со всеми признаками
    current_features = list(initial_features)
    current_score = evaluate_model_with_cv(
        X_original[current_features], y_original,
        [f for f in categorical_features if f in current_features],
        model_params, cv_strategy
    )
    
    best_features = list(current_features)
    best_score = current_score
    print(f"Initial score with all features ({len(current_features)}): {current_score:.4f}")
    
    rounds_since_improvement = 0
    num_features_removed = 0

    # Будем удалять признаки до тех пор, пока не останется 1 признак
    # или пока не достигнем max_features_to_remove
    while len(current_features) > 1 and \
          (max_features_to_remove is None or num_features_removed < max_features_to_remove):
        
        scores_if_removed = {} # Словарь: признак -> метрика без него
        
        # Для каждого признака, который можно потенциально удалить в текущем шаге
        for feature_to_remove in current_features:
            temp_features = [f for f in current_features if f != feature_to_remove]
            
            # Если после удаления этого признака останется 0 признаков, пропускаем
            if not temp_features:
                continue

            # Оцениваем модель без этого признака
            score = evaluate_model_with_cv(
                X_original[temp_features], y_original,
                [f for f in categorical_features if f in temp_features],
                model_params, cv_strategy
            )
            scores_if_removed[feature_to_remove] = score
        
        if not scores_if_removed: # Если current_features состоит из 1 элемента, то scores_if_removed будет пуст
            break

        # Находим признак, удаление которого дает наименьшее ухудшение (или наибольшее улучшение)
        # Для LogLoss: ищем признак, удаление которого МИНИМИЗИРУЕТ значение LogLoss
        feature_to_eliminate = min(scores_if_removed, key=scores_if_removed.get)
        
        # Скор, если мы удалим этот признак
        candidate_score = scores_if_removed[feature_to_eliminate]

        print(f"  Candidate for removal: '{feature_to_eliminate}'. Score if removed: {candidate_score:.4f} (current features: {len(current_features)})")

        # Обновляем текущий набор признаков (удаляем выбранный признак)
        current_features.remove(feature_to_eliminate)
        num_features_removed += 1

        # Проверяем, улучшилась ли метрика на этом шаге
        if candidate_score < best_score: # LogLoss: чем меньше, тем лучше
            best_score = candidate_score
            best_features = list(current_features) # Сохраняем этот набор как лучший
            rounds_since_improvement = 0 # Сбрасываем счетчик, т.к. было улучшение
            print(f"  -> IMPROVEMENT! New best score: {best_score:.4f}. Features remaining: {len(current_features)}. This is now the best set.")
        else:
            rounds_since_improvement += 1
            print(f"  -> No improvement. Current score: {candidate_score:.4f}. Rounds since improvement: {rounds_since_improvement}")
            if rounds_since_improvement >= patience_rounds:
                print(f"  Stopping: No improvement for {patience_rounds} rounds.")
                break # Останавливаемся, если метрика не улучшается достаточно долго
            
    print(f"\nFeature selection complete. Best score: {best_score:.4f} with {len(best_features)} features.")
    return best_features, best_score

# --- Objective функция для Optuna ---
def objective_custom_rfe_optimized(trial):
    # Гиперпараметры CatBoost
    cat_params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-4, 10.0),
        'random_seed': 42,
        'verbose': 0,
        'eval_metric': 'Logloss',
        'early_stopping_rounds': 50
    }
    
    # Гиперпараметры для алгоритма отбора признаков
    # Можно ограничить максимальное количество удаляемых признаков
    max_features_to_remove = trial.suggest_int('max_features_to_remove', 0, X.shape[1] - 1)
    patience_rounds = trial.suggest_int('patience_rounds', 1, 5)

    # Внешняя кросс-валидация для оценки всего процесса
    cv_strategy_outer = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    total_oof_preds = []
    total_oof_y = []

    for fold, (train_idx, val_idx) in enumerate(cv_strategy_outer.split(X, y)):
        X_train_fold, X_val_fold = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
        y_train_fold, y_val_fold = y.iloc[train_idx].copy(), y.iloc[val_idx].copy()

        print(f"\n--- Fold {fold+1} / {cv_strategy_outer.n_splits} ---")
        
        # Внутренняя CV для recursive_feature_selection_by_metric
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42 + fold)

        # Шаг 1: Отбор признаков на обучающей части текущего фолда
        selected_features_fold, _ = recursive_feature_selection_by_metric(
            X_train_fold, y_train_fold, categorical_features,
            cat_params, # Передаем параметры для CatBoost, который будет внутри отбора
            inner_cv,
            max_features_to_remove=max_features_to_remove,
            patience_rounds=patience_rounds
        )
        
        if not selected_features_fold: # Если не выбрано ни одного признака, это плохо
            return float('inf')

        # Шаг 2: Обучение финальной модели на выбранных признаках (и на оригинальных данных с пропусками)
        X_train_final = X_train_fold[selected_features_fold]
        X_val_final = X_val_fold[selected_features_fold]

        final_cat_features_for_model = [col for col in selected_features_fold if col in categorical_features]

        # Создаем Pool для финальной модели
        train_pool_final = Pool(X_train_final, y_train_fold, cat_features=final_cat_features_for_model)
        val_pool_final = Pool(X_val_final, y_val_fold, cat_features=final_cat_features_for_model)
        
        final_model_fold = CatBoostClassifier(**cat_params) # Новая модель для каждого фолда
        final_model_fold.fit(train_pool_final, eval_set=val_pool_final, early_stopping_rounds=cat_params['early_stopping_rounds'], verbose=0)
        
        oof_preds_fold = final_model_fold.predict_proba(X_val_final)[:, 1]
        
        total_oof_preds.extend(oof_preds_fold)
        total_oof_y.extend(y_val_fold)
        
        # Отчет для Optuna (чтобы она могла обрезать неперспективные пробные версии)
        # Важно: reporting the metric for the current fold, not the aggregated one.
        trial.report(log_loss(y_val_fold, oof_preds_fold), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    final_metric_value = log_loss(total_oof_y, total_oof_preds)
    return final_metric_value

# --- Запуск Optuna исследования ---
# Для демонстрации n_trials=5, в реальном проекте - намного больше
study = optuna.create_study(direction='minimize', study_name='recursive_feature_selection_by_metric_optimized')
study.optimize(objective_custom_rfe_optimized, n_trials=5) 

print("\n--- Оптимизация завершена ---")
print("Лучшие параметры:", study.best_params)
print("Лучшее значение (LogLoss):", study.best_value)
