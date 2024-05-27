import pandas as pd
import redis

from catboost import CatBoostClassifier

from reaction_predicter.catboost_train.train import fit_model
from reaction_predicter.features.prepare_test_feature import build_dataset


def get_probability_one(
        reactive: list,
        product: str,
):
    def get_redis_connection():
        redis_client = redis.Redis(host = 'redis', port=6379, db=0)
        return redis_client
    
    redis_client = get_redis_connection()
    
    model, linear_model = fit_model(redis_client, product)
    feature = build_dataset(reactive, redis_client)

    proba_boost = model.predict_proba(feature)[0, -1]
    proba = linear_model.predict_proba(pd.DataFrame([proba_boost]))[0, -1]
    proba = f'{round(proba, 2) * 100}%'

    return {"reaction": f"{' + '.join(reactive)} >> {product}", 'proba': proba}