# обучение модели
import random
from fastapi import Depends

from reaction_predicter.catboost_train.prepare_train_dataset import DataSet

def tipo_obuchenie(redis_client):

    dataset = DataSet(redis_client)
    return dataset.feature_prepare().shape[0]