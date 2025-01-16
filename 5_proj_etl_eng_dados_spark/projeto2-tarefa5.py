# Projeto 2 - Banco de Dados, Machine Learning e Pipeline ETL em Cluster Spark Para Detectar Anomalias em Transações Financeiras
# Tarefa 5 - Tudo Junto - Machine Learning, Engenharia de Dados, ETL e Banco de Dados no Mesmo Pipeline

# Imports
import os
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import warnings
warnings.filterwarnings('ignore')

# Inicializando a sessão Spark
spark = SparkSession.builder \
    .appName("DSA Projeto 2 - Tarefa 5") \
    .getOrCreate()

# Configura o nível de log
spark.sparkContext.setLogLevel('ERROR')

# Carregar dados
df_dsa = spark.read.csv('data/dados1_cap05.csv', header = True, inferSchema = True)

# Extração de características numéricas da coluna de data
df_dsa = df_dsa.withColumn("Ano", year(df_dsa["Data"]))
df_dsa = df_dsa.withColumn("Mes", month(df_dsa["Data"]))
df_dsa = df_dsa.withColumn("Dia", dayofmonth(df_dsa["Data"]))
df_dsa = df_dsa.withColumn("Hora", hour(df_dsa["Data"]))

# Selecionando características relevantes
features = ['Valor', 'Ano', 'Mes', 'Dia', 'Hora']

# VectorAssembler para combinar características
assembler = VectorAssembler(inputCols=features, outputCol="features")
df_dsa = assembler.transform(df_dsa)

# Normalizar os dados
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
df_dsa = scaler.fit(df_dsa).transform(df_dsa)

# Cria o modelo KMeans
kmeans = KMeans(featuresCol='scaledFeatures', k=3)  

# Treinar o modelo KMeans
modelo = kmeans.fit(df_dsa)

# Atribuir pontos aos clusters
df_resultado = modelo.transform(df_dsa)

# Supondo que você queira saber a contagem de pontos em cada cluster
cluster_counts = df_resultado.groupBy('prediction').count()
cluster_counts.show()

# Você também pode remover a coluna 'features' ou 'scaledFeatures' se não forem mais necessárias
df_resultado_clean = df_resultado.drop('features', 'scaledFeatures')
df_resultado_clean.show()

# Contagem total de linhas no DataFrame 
print(f"Total de linhas no DataFrame: {df_resultado_clean.count()}")

# Convertendo para DataFrame do Pandas
pandas_df = df_resultado_clean.toPandas()

# Caminho da pasta onde o banco de dados será salvo
output_dir = 'data/resultado_tarefa5'

# Verificar se a pasta existe; se não, criar
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Salvar o DataFrame do Pandas em uma nova tabela no SQLite
conn = sqlite3.connect("data/resultado_tarefa5/dsa_database_ml.db")
pandas_df.to_sql("tb_clusters", conn, if_exists="replace", index=False)
conn.close()

# Encerrando a sessão Spark
spark.stop()

print("\nObrigado DSA. Execução do Job Concluída com Sucesso!\n")

