# готовим таргет для обучения
import boto3
import pandas as pd
import io
import redis
import json

from reaction_predicter.features.rdkit_description.rdkit_features import ECFP6

##подключение к пакету

class DataSet:
    def __init__(self, redis_client):

        feature_table_name = 'data_reaction'
        target_table_name = 'data_product'

        cached_table = redis_client.get(feature_table_name)

        if cached_table is not None:
            # Если таблица есть в кэше, возвращаем её
            cached_table_json = cached_table.decode('utf-8')
            cached_data = json.loads(cached_table_json)
            self.dataframe = pd.DataFrame(cached_data)
        else:
            session = boto3.session.Session()
            s3 = session.client(
                service_name='s3',
                endpoint_url='https://storage.yandexcloud.net'
                )

            data_reactive = s3.get_object(Bucket='data-proc-baket',Key='dataframe_10k.tsv')['Body'].read().decode('utf-8')
            self.df_reactive = pd.read_table(io.StringIO(data_reactive))
            redis_client.set(feature_table_name, self.df_reactive.to_json())

            data_product = s3.get_object(Bucket='data-proc-baket',Key='vec_prod_10k.tsv')['Body'].read().decode('utf-8')
            self.df_product = pd.read_table(io.StringIO(data_product))
            redis_client.set(target_table_name, self.df_product.to_json())

    @staticmethod
    def tanimoto_similarity(list1, list2):
        intersection = sum(x and y for x, y in zip(list1, list2))
        union = sum(x or y for x, y in zip(list1, list2))
        koef = intersection/union if union != 0 else 0
        
        return koef

    def target_prepare(self, reqest_product):
        ecfp6 = ECFP6([reqest_product])
        vec_reqest_product = list(ecfp6.compute_ECFP6('smiles', nBits = 512).drop('smiles', axis = 1).iloc[0])

        target = []
        for i in self.df_product.shape[0]:
            target.append(self.tanimoto_similarity(list(self.df_product.iloc[i]), vec_reqest_product))

        series_target = pd.Series(target)
        q90 = series_target.quantile(0.90)
        binary_target = series_target.apply(lambda x: 1 if x >= q90 else 0)

        return binary_target

    def feature_prepare(self):
        return self.dataframe
