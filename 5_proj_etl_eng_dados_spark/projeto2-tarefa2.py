# Projeto 2 - Banco de Dados, Machine Learning e Pipeline ETL em Cluster Spark Para Detectar Anomalias em Transações Financeiras
# Tarefa 2 - Criando o Segundo Pipeline Para Engenharia de Dados de Transações Financeiras

# Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import pandas as pd

# Inicializando a sessão Spark
spark = SparkSession.builder \
    .appName("DSA Projeto 2 - Tarefa 2") \
    .getOrCreate()

# Carregando dados de um arquivo CSV
df_dsa = spark.read.csv('data/dados1_cap05.csv', header = True, inferSchema = True)

# Configurando o nível de log
spark.sparkContext.setLogLevel('ERROR')

# Agregação - média de valor transação por ID_Cliente
df_dsa_dados_agregados = df_dsa.groupBy("ID_Cliente").agg(avg("Valor").alias("Media_Transacao"))

# Contagem total de linhas no DataFrame de anomalias
print(f"\nTotal de linhas no DataFrame df_dsa_dados_agregados: {df_dsa_dados_agregados.count()}")

# Salvando o resultado em um arquivo CSV
df_dsa_dados_agregados.write.mode('overwrite').parquet('data/resultado_tarefa2')

# Encerrando a sessão Spark
spark.stop()

print("\nObrigado DSA. Execução do Job Concluída com Sucesso!\n")
