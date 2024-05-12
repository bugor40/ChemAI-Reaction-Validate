# обучение модели
import random
import redis
from fastapi import Depends

from reaction_predicter.catboost_train.prepare_train_dataset import DataSet

def tipo_obuchenie():

    def get_redis_connection():
        redis_client = redis.Redis(host = 'redis', port=6379, db=0)
        return redis_client

    dataset = DataSet(redis_client = get_redis_connection())
    return dataset.feature_prepare().shape[0]