import time
import pandas as pd
import redis

from reaction_predicter.catboost_train.train import tipo_obuchenie
from reaction_predicter.features.prepare_test_feature import build_dataset


def get_probability_one(
        reactive: list,
        product: str,
):
    def get_redis_connection():
        redis_client = redis.Redis(host = 'redis', port=6379, db=0)
        return redis_client
    
    redis_client = get_redis_connection()
    
    model = tipo_obuchenie(redis_client)
    feature = build_dataset(reactive, redis_client)

    print(model)
    print(feature)

    proba = feature.shape

    time.sleep(10)
    return {"reaction": f"{'+'.join(reactive)} = {product}", 'proba': proba}