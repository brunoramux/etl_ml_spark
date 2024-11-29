# Projeto 3 - Pipeline de Limpeza e Transformação Para Aplicações de IA com PySpark SQL

# Imports
import os
import pyspark
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import pyspark.ml.evaluation as evals
import pyspark.ml.tuning as tune
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.feature import  VectorAssembler
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.sql.functions import round, desc

spark = SparkSession.builder \
    .appName('Projeto3-Exp') \
    .master('yarn') \
    .config('spark.submit.deployMode', 'client') \
    .config('spark.driver.memory', '4g') \
    .config('spark.executor.memory', '1g') \
    .config('spark.executor.cores', '2') \
    .getOrCreate()
    
spark.sparkContext.setLogLevel("ERROR")
    
df_dsa_aeroportos = spark.read.csv("/opt/spark/data/dataset1.csv", header = True)

df_dsa_aeroportos.show(10)

print("Job finalizado")
