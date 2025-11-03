def process_client(args):
    client_id, client_data_dict, target_dates = args
    
    # Восстанавливаем DataFrame из dict (multiprocessing не передаёт pandas напрямую хорошо)
    client_data = pd.DataFrame(client_data_dict)
    client_data['EV_SYS_RGSTRN_DTTM'] = pd.to_datetime(client_data['EV_SYS_RGSTRN_DTTM'])
    
    rows = []
    for target_date in target_dates:
        # История: последние 180 дней ДО target_date
        hist = client_data[
            (client_data['EV_SYS_RGSTRN_DTTM'] > target_date - pd.Timedelta(days=180)) &
            (client_data['EV_SYS_RGSTRN_DTTM'] <= target_date)
        ]
        if hist.empty:
            continue

        # Будущие платежи: в течение 5 дней ПОСЛЕ target_date (для таргета)
        future = client_data[
            (client_data['EV_SYS_RGSTRN_DTTM'] > target_date) &
            (client_data['EV_SYS_RGSTRN_DTTM'] <= target_date + pd.Timedelta(days=5))
        ]

        # Агрегация по группам
        agg = hist.groupby('GROUP_ID').agg(
            TOTAL_AMOUNT=('EV_TSACTN_AMT', 'sum'),
            PAYMENT_COUNT=('EV_ID', 'count'),
            LAST_PAYMENT_DATE=('EV_SYS_RGSTRN_DTTM', 'max')
        ).reset_index()

        if agg.empty:
            continue

        # Признаки
        agg['DAYS_SINCE_LAST'] = (target_date - agg['LAST_PAYMENT_DATE']).dt.days
        agg['RANK_SUM'] = agg['TOTAL_AMOUNT'].rank(method='min', ascending=False)
        agg['RANK_FREQ'] = agg['PAYMENT_COUNT'].rank(method='min', ascending=False)
        agg['RANK_RECENCY'] = agg['DAYS_SINCE_LAST'].rank(method='min', ascending=True)

        # RRF скоры
        k = 60
        agg['SUM_SCORE'] = 1 / (agg['RANK_SUM'] + k)
        agg['FREQ_SCORE'] = 1 / (agg['RANK_FREQ'] + k)
        agg['RECENCY_SCORE'] = 1 / (agg['RANK_RECENCY'] + k)

        # Таргет: была ли эта группа в будущих платежах?
        future_groups = set(future['GROUP_ID'].unique())
        agg['TARGET'] = agg['GROUP_ID'].isin(future_groups).astype(int)

        agg['CLIENT_ID'] = client_id
        agg['TARGET_DATE'] = target_date
        rows.append(agg)

    if rows:
        result = pd.concat(rows, ignore_index=True)
        # Возвращаем как dict для сериализации
        return result.to_dict('records')
    else:
        return []


def run_parallel(data, target_dates, n_jobs=None):
    if n_jobs is None:
        n_jobs = min(cpu_count(), 8)  # не больше 8, чтобы не убить память

    # Группируем по клиентам
    client_groups = list(data.groupby('CLIENT_ID'))
    
    # Подготавливаем аргументы: (client_id, client_data_dict, target_dates)
    tasks = []
    for client_id, group_df in client_groups:
        tasks.append((client_id, group_df.to_dict('list'), target_dates))

    print(f"Запуск обработки {len(tasks)} клиентов на {n_jobs} ядрах...")

    with Pool(processes=n_jobs) as pool:
        results = pool.map(process_client, tasks)

    # Собираем результат
    all_rows = []
    for res in results:
        if res:
            all_rows.extend(res)

    final_df = pd.DataFrame(all_rows)
    return final_df


# Ограничим клиентов для теста (уберите после)
# test_clients = data['CLIENT_ID'].unique()[:1000]
# data = data[data['CLIENT_ID'].isin(test_clients)]

final_df = run_parallel(data, target_dates, n_jobs=6)

# Сохраняем
final_df.to_parquet("ml_train_data.parquet", index=False)
print("Готово! Размер:", final_df.shape)
