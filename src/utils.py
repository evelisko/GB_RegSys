import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import ItemItemRecommender

def get_freq_encoder(data,feature_names):
    for feature_name in feature_names:
        freq_encoder = data[feature_name].value_counts(normalize=True)
        data[feature_name] = data[feature_name].map(freq_encoder)
    return data

def category_to_digit(df, features):
    df = df.copy(deep=True)
    for i, feature in enumerate(features):
        # feature = str.replace(feature,' ','_')
        values_list = df[feature].value_counts()
        names = sorted(values_list.index)
        # names = sorted(names)
        for name in names:
            name = str.replace(name,' ','_')
            df.insert(3, f'{feature}_{name}', np.where((df[feature]==name),1,0), True)
    df.drop(features, axis=1, inplace=True)
    return df
    

def popularity_recommendation(data, n=5):
    """Топ-n популярных товаров"""
    popular = data.groupby('item_id')['sales_value'].sum().reset_index()
    popular.sort_values('sales_value', ascending=False, inplace=True)
    recs = popular.head(n).item_id
    return recs.tolist()


def prefilter_items(data, item_features, drop_categories=[],take_n_popular=5000):
    # 1. Уберем товары, которые не продавались за последние 12 месяцев
    data = data.loc[~(data['week_no'] < data['week_no'].max() - 12)]

    # 2. Уберем не интересные для рекоммендаций категории (department)
    not_important_goods = item_features.loc[(item_features['department'].isin(drop_categories)), 'item_id'].tolist()
    data = data.loc[(~data['item_id'].isin(not_important_goods))]

    # 3. Уберем слишком дешевые товары (на них не заработаем). Товары, со средней ценой < 1$
    data.drop(data[data['sales_value'] < 1].index, axis=0, inplace=True)

    # 4. Уберем слишком дорогие товары. Товары со средней ценой > 30$
    data.drop(data[data['sales_value'] > 30].index, axis=0, inplace=True)

    # 5. Уберем самые популярные товары (их и так купят)
    popularity = data.groupby('item_id')['user_id'].nunique().reset_index() / data['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.8].item_id.tolist()
    data = data.loc[(~data['item_id'].isin(top_popular))]
    # data = data.loc[(~data['item_id'].isin(not_important_goods))]
    #
    # # 6. Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data = data.loc[(~data['item_id'].isin(top_notpopular))]
    # result = data
	
    # Возьмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()	
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    # ...
    return data

#---------------------------------------------------------------------------------------------------
def postfilter_items(recommendations, item_features, N=5):
    """Пост-фильтрация товаров
    
    Input
    -----
    recommendations: list
        Ранжированный список item_id для рекомендаций
    item_info: pd.DataFrame
        Датафрейм с информацией о товарах
    """
    
    # Уникальность
#     recommendations = list(set(recommendations)) - неверно! так теряется порядок
    unique_recommendations = []
    [unique_recommendations.append(item) for item in recommendations if item not in unique_recommendations]
    
    # Разные категории
    categories_used = []
    final_recommendations = []
    
    CATEGORY_NAME = 'sub_commodity_desc'
    for item in unique_recommendations:
        category = item_features.loc[item_features['item_id'] == item, CATEGORY_NAME].values[0]
        
        if category not in categories_used:
            final_recommendations.append(item)
            
        unique_recommendations.remove(item)
        categories_used.append(category)
    
    # Для каждого юзера 5 рекомендаций (иногда модели могут возвращать < 5)
    n_rec = len(final_recommendations)
    if n_rec < N:
        final_recommendations.extend(unique_recommendations[:N - n_rec])  # (!) это не совсем верно
    else:
        final_recommendations = final_recommendations[:N]
        
    # 2 новых товара (юзер никогда не покупал)
    # your_code
    
    # 1 дорогой товар, > 7 долларов
    # your_code
    
    assert len(final_recommendations) == N, 'Количество рекомендаций != {}'.format(N)
    return final_recommendations
#------------------------------------------------------------------------------------------------

def get_similar_item(model,  itemid_to_id, id_to_itemid, x):
    id = itemid_to_id[x]
    recs = model.similar_items(id, N=2)
    top_rec = recs[1][0]
    return id_to_itemid[top_rec]


def get_similar_items_recommendation(user, data, itemid_to_id, id_to_itemid,  model, N=5):
    """Рекомендуем товары, похожие на топ-N купленных юзером товаров"""

    top_purchases = data.groupby(['user_id', 'item_id'])['quantity'].count().reset_index()
    top_purchases.sort_values('quantity', ascending=False, inplace=True)
    top_purchases = top_purchases[top_purchases['item_id'] != 999999]

    top_users_purchases = top_purchases[top_purchases['user_id'] == user].head(N)
    res = top_users_purchases['item_id'].apply(lambda x: get_similar_item(model, itemid_to_id=itemid_to_id, id_to_itemid=id_to_itemid, x=x)).tolist()
    return res

def fit_own_recomender(user_item_matrix):
    own = ItemItemRecommender(K=1, num_threads=4) # K - кол-во билжайших соседей
    own.fit(csr_matrix(user_item_matrix).T.tocsr(), show_progress=False)
    return own

def get_own_recommendations(own, userid, user_item_matrix, N):
    recs = own.recommend(userid=userid, 
                        user_items=csr_matrix(user_item_matrix).tocsr(),   # на вход user-item matrix
                        N=N, 
                        filter_already_liked_items=False, 
                        filter_items=None, 
                        recalculate_user=False)
    return recs

def get_similar_users_recommendation(userid, userid_to_id, id_to_userid, user_item_matrix, model, N=5):
    """Рекомендуем топ-N товаров, среди купленных похожими юзерами"""
    res = []

    # Находим топ-N похожих пользователей
    similar_users = model.similar_users(userid_to_id[userid], N=N+1) # user + N его друзей.
    similar_users = [rec[0] for rec in similar_users]
    similar_users = similar_users[1:]   # удалим юзера из запроса
 
    own = fit_own_recomender(user_item_matrix)
 
    for user in similar_users:
        userid = id_to_userid[user] #own recommender works with user_ids
        res.extend(get_own_recommendations(own, userid, user_item_matrix, N=1))

    return res

#--------------------------------------------------------------------------------------------------------
def perpare_lvl2_1(val_data, train_data, recommender, item_features, user_features, N=50):
    # val_data = data_train_lvl_2.copy()
    # train_data = data_train_lvl_1.copy()

    users_warm = pd.DataFrame(val_data['user_id'].unique()) # Добавим туда еще фитчи user-ов и item-ов.
    users_warm.columns = ['user_id']
    # Пока только warm start
    users_warm = users_warm[users_warm['user_id'].isin(train_data['user_id'].unique())]

    users_cold = pd.DataFrame(val_data['user_id'].unique()) # Добавим туда еще фитчи user-ов и item-ов.
    users_cold.columns = ['user_id']
    # cold_start
    users_cold = users_cold[~users_cold['user_id'].isin(users_warm['user_id'].unique())]

    # Заполняем кандидатов, на основе предсказания модели 1-го уровня.
    users_cold['candidates'] = users_cold['user_id'].apply(lambda x: recommender.get_top_popular(N=N))
    s = users_cold.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'item_id'

    # Это кандидаты. (т.е. предпологаемые покупки совершенные на основе предсказаний.)
    users_cold = users_cold.drop('candidates', axis=1).join(s)
    users_cold['drop'] = 1  # фиктивная переменная
    # Заполняем кандидатов, на основе предсказания модели 1-го уровня.
    users_warm['candidates'] = users_warm['user_id'].apply(lambda x: recommender.get_own_recommendations(x, N=N))
    # test_users = data
    s = users_warm.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'item_id'

    # Это кандидаты. (т.е. предпологаемые покупки совершенные на основе предсказаний.)
    users_warm = users_warm.drop('candidates', axis=1).join(s)
    users_warm['drop'] = 1  # фиктивная переменная

    # Создадим таблицу с реальными покупками user-ов. 
    targets = val_data[['user_id', 'item_id']].copy() # свойства 
    targets['target'] = 1  # тут только покупки

    # Объединим предпологаемые покупки с реальными, совершенными user-ами.
    targets_cold = users_cold.merge(targets, on=['user_id', 'item_id'], how='left')

    # В результате, напротив товаров, в редсказании которых мы ошиблись, 
    # будет стоять Nan. Заполним их  нулями.  
    targets_cold['target'].fillna(0, inplace= True)
    targets_cold.drop('drop', axis=1, inplace=True)
    # Добавим к нашему датасету фичи user-ов и item-ов.
    targets_cold = targets_cold.merge(item_features, on='item_id', how='left')
    targets_cold = targets_cold.merge(user_features, on='user_id', how='left')

    # Объединим предпологаемые покупки с реальными, совершенными user-ами.
    targets_warm = users_warm.merge(targets, on=['user_id', 'item_id'], how='left')

    # В результате, напротив товаров, в редсказании которых мы ошиблись, 
    # будет стоять Nan. Заполним их  нулями.  
    targets_warm['target'].fillna(0, inplace= True)
    targets_warm.drop('drop', axis=1, inplace=True)
    # targets_warm['target'].mean() #Угадали примерно 17% покупок.

    # Добавим к нашему датасету фичи user-ов и item-ов.
    targets_warm = targets_warm.merge(item_features, on='item_id', how='left')
    targets_warm = targets_warm.merge(user_features, on='user_id', how='left')

    targets_lvl_2 = pd.concat([targets_warm, targets_cold], ignore_index=True)

    X_ = targets_lvl_2.drop('target', axis=1)
    y_ = targets_lvl_2[['target']]

    return X_, y_,

#------------------------------------------------------------------------------------------------

def perpare_lvl2(val_data, train_data, recommender, item_features, user_features, N=50):
    users_lvl_2 = pd.DataFrame(val_data['user_id'].unique()) # Добавим туда еще фитчи user-ов и item-ов.
    users_lvl_2.columns = ['user_id']

    # Пока только warm start
    users_lvl_2 = users_lvl_2[users_lvl_2['user_id'].isin(train_data['user_id'].unique())]

    # Заполняем кандидатов, на основе предсказания модели 1-го уровня.
    users_lvl_2['candidates'] = users_lvl_2['user_id'].apply(lambda x: recommender.get_own_recommendations(x, N=N))
    # test_users = data
    s = users_lvl_2.apply(lambda x: pd.Series(x['candidates']), axis=1).stack().reset_index(level=1, drop=True)
    s.name = 'item_id'

    # Это кандидаты. (т.е. предпологаемые покупки совершенные на основе предсказаний.)
    users_lvl_2 = users_lvl_2.drop('candidates', axis=1).join(s)
    users_lvl_2['drop'] = 1  # фиктивная переменная

    # Создадим таблицу с реальными покупками user-ов. 
    targets_lvl_2 = val_data[['user_id', 'item_id']].copy() # свойства 
    targets_lvl_2['target'] = 1  # тут только покупки 

    # Объединим предпологаемые покупки с реальными, совершенными user-ами.
    targets_lvl_2 = users_lvl_2.merge(targets_lvl_2, on=['user_id', 'item_id'], how='left')

    # В результате, напротив товаров, в редсказании которых мы ошиблись, 
    # будет стоять Nan. Заполним их  нулями.  
    targets_lvl_2['target'].fillna(0, inplace= True)
    targets_lvl_2.drop('drop', axis=1, inplace=True)
    targets_lvl_2['target'].mean() #Угадали примерно 17% покупок.

    # Добавим к нашему датасету фичи user-ов и item-ов.
    targets_lvl_2 = targets_lvl_2.merge(item_features, on='item_id', how='left')
    targets_lvl_2 = targets_lvl_2.merge(user_features, on='user_id', how='left')

    X_ = targets_lvl_2.drop('target', axis=1)
    y_ = targets_lvl_2[['target']]
    return X_, y_