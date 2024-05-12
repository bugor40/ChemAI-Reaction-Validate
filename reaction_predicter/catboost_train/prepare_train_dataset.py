# готовим таргет для обучения
import boto3
import pandas as pd
import io
import redis
import json

##подключение к пакету

class DataSet:
    def __init__(self, redis_client):
        feature_table_name = 'data_reaction'
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

            data = s3.get_object(Bucket='data-proc-baket',Key='dataframe_10k.tsv')['Body'].read().decode('utf-8')
            self.dataframe = pd.read_table(io.StringIO(data))
            redis_client.set(feature_table_name, self.dataframe.to_json())

    def target_prepare(self):
        pass

    def feature_prepare(self):
        return self.dataframe
