import time
import pandas as pd

from reaction_predicter.catboost_train.train import tipo_obuchenie
from reaction_predicter.features.prepare_test_feature import build_dataset


def get_probability_one(
        reactive: list,
        product: str,
):
    
    model = tipo_obuchenie()
    feature = build_dataset(reactive)

    print(model)
    print(feature)
    

    proba = model

    time.sleep(10)
    return {"reaction": f"{'+'.join(reactive)} = {product}", 'proba': proba}