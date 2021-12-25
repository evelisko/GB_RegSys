import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix
from implicit.nearest_neighbours import ItemItemRecommender

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
	
    # Возбмем топ по популярности
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()	
    
    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999
    
    # ...
    return data


def postfilter_items():
    pass

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