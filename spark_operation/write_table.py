from pyspark.sql import SparkSession
# from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
from datetime import datetime
import pandas as pd


spark = SparkSession.builder\
        .master("local[*]")\
        .appName("write_data")\
        .getOrCreate()

df = pd.read_pickle('/home/ubuntu/data/dataframe_100k.pickle')

df = spark.createDataFrame(df)
df.write.mode("overwrite").save(f"./data_reaction", format="parquet")

spark.stop()