import os
import json
import pandas as pd
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassificationModel

def init_context(context):
    # Spark
    conf = SparkConf()\
        .setMaster("local[*]")\
        .set("spark.cores.max", "2")\
        .setAppName("Model Inferencing")
    context.sc = SparkContext.getOrCreate(conf=conf)
    context.spark = SQLContext(context.sc)
    
    # Model - path should be in format of file:///User/admin/....
    context.model = RandomForestClassificationModel.load(os.getenv('MODEL_PATH'))
    
    # Vector Assembler
    context.assembler = VectorAssembler(inputCols=['param1', 'param2', 'param3'], outputCol="features")

def handler(context, event):
    # Transform incoming features
    pandas_df = pd.DataFrame([event.body])
    spark_df = context.spark.createDataFrame(pandas_df)
    features = context.assembler.transform(spark_df).select("features")
    
    # Model prediction
    pred = context.model.transform(features)
    
    # Return response in JSON format
    return pred.toJSON().first()