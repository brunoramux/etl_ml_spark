# Projeto 2 - Banco de Dados, Machine Learning e Pipeline ETL em Cluster Spark Para Detectar Anomalias em Transações Financeiras
# Tarefa 3 - Ajustando o Pipeline ETL Para Carregar o Resultado Direto em Banco de Dados

# Imports
import os
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg

# Inicializando a sessão Spark
spark = SparkSession.builder \
    .appName("DSA Projeto 2 - Tarefa 3") \
    .getOrCreate()

# Carregando dados de um arquivo CSV
df_dsa = spark.read.csv('data/dados1_cap05.csv', header = True, inferSchema = True)

# Configurando o nível de log
spark.sparkContext.setLogLevel('ERROR')

# Agregação - média de valor transação por ID_Cliente
df_dsa_dados_agregados = df_dsa.groupBy("ID_Cliente").agg(avg("Valor").alias("Media_Transacao"))

# Convertendo para DataFrame do Pandas
pandas_df = df_dsa_dados_agregados.toPandas()

# Caminho da pasta onde o banco de dados será salvo
output_dir = 'data/resultado_tarefa3'

# Verificar se a pasta existe; se não, criar
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Encerrando o DataFrame do Pandas em uma nova tabela no SQLite
conn = sqlite3.connect("data/resultado_tarefa3/dsa_database.db")
pandas_df.to_sql("tb_anomalias_agregadas", conn, if_exists = "replace", index = False)
conn.close()

# Encerrar a sessão Spark
spark.stop()

print("\nObrigado DSA. Execução do Job Concluída com Sucesso!\n")


