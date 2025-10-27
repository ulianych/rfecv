def calculate_6month_ranks_optimized(df):
    """
    Оптимизированная версия с предварительной подготовкой данных
    """
    
    # Сортируем и создаем копию
    df_sorted = df.sort_values(['client_id', 'payment_date']).copy()
    features = ['total_6month', 'count_6month', 'avg_6month', 'min_6month', 'max_6month']
    
    # Создаем мультииндекс для быстрого поиска
    df_sorted['date_index'] = df_sorted['payment_date']
    df_sorted.set_index(['client_id', 'date_index'], inplace=True)
    df_sorted.sort_index(inplace=True)
    
    results = []
    
    # Обрабатываем каждого клиента отдельно
    for client_id in df_sorted.index.get_level_values(0).unique():
        client_data = df_sorted.xs(client_id, level=0)
        
        # Для каждой строки клиента
        for idx, row in client_data.iterrows():
            current_date = row['payment_5_days_ago']
            payment_group = row['payment_group']
            
            if pd.isna(current_date):
                ranks = {f'{feat}_rank': np.nan for feat in features}
                ranks.update({
                    'client_id': client_id,
                    'payment_group': payment_group,
                    'payment_date': row['payment_date']
                })
                results.append(ranks)
                continue
            
            # Используем индекс для быстрого поиска в окне
            window_start = current_date - np.timedelta64(180, 'D')
            
            try:
                # Быстрый поиск по индексу
                window_data = client_data.loc[window_start:current_date - np.timedelta64(1, 'D')]
            except:
                window_data = pd.DataFrame()
            
            if len(window_data) == 0:
                ranks = {f'{feat}_rank': np.nan for feat in features}
                ranks.update({
                    'client_id': client_id,
                    'payment_group': payment_group, 
                    'payment_date': row['payment_date']
                })
                results.append(ranks)
                continue
            
            # Агрегируем данные
            aggregated = window_data.groupby('payment_group').agg({
                'total_6month': 'sum',
                'count_6month': 'sum',
                'avg_6month': 'mean',
                'min_6month': 'min', 
                'max_6month': 'max'
            }).reset_index()
            
            # Вычисляем ранги
            ranks = {}
            for feat in features:
                valid_data = aggregated[aggregated[feat].notna() & (aggregated[feat] != 0)]
                
                if len(valid_data) == 0:
                    ranks[f'{feat}_rank'] = np.nan
                    continue
                
                valid_data = valid_data.copy()
                valid_data['rank'] = valid_data[feat].rank(method='dense', ascending=False)
                
                current_rank = valid_data[valid_data['payment_group'] == payment_group]
                ranks[f'{feat}_rank'] = current_rank['rank'].iloc[0] if len(current_rank) > 0 else np.nan
            
            ranks.update({
                'client_id': client_id,
                'payment_group': payment_group,
                'payment_date': row['payment_date']
            })
            results.append(ranks)
    
    # Сбрасываем индекс для объединения
    df_sorted.reset_index(inplace=True)
    ranks_df = pd.DataFrame(results)
    
    result_df = df_sorted.merge(ranks_df, on=['client_id', 'payment_group', 'payment_date'], how='left')
    
    # Удаляем временные колонки
    result_df.drop('date_index', axis=1, inplace=True)
    
    return result_df

# Использование оптимизированной версии
df_with_ranks = calculate_6month_ranks_optimized(df_complete)
