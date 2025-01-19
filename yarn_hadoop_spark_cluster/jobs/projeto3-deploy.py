# Projeto 3 - Pipeline de Machine Learning em Cluster Spark Para Previsão de Churn - Treinamento e Deploy
# Script de Deploy do Modelo

# Imports
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Inicializando a Spark Session
spark = SparkSession.builder.appName("Projeto3-Deploy").getOrCreate()

# Carregando o modelo do HDFS 
modelo_dsa = PipelineModel.load("hdfs:///opt/spark/data/modelo")

# Carregando novos dados para previsão (coloque o arquivo no HDFS como mostrado nas aulas)
novos_dados = spark.read.csv("hdfs:///opt/spark/data/novosdados.csv", header=True, inferSchema=True)

# Aplicando o modelo aos novos dados para fazer previsões
# Isso inclui as etapas de StringIndexer e VectorAssembler (pipeline)
previsoes = modelo_dsa.transform(novos_dados)

# Selecionando as colunas necessárias para salvar no HDFS
selected_columns = ["Idade", "UsoMensal", "Plano", "SatisfacaoCliente", "TempoContrato", "ValorMensal", "prediction"]
previsoes_para_salvar = previsoes.select(selected_columns)

# Salvando no HDFS em modo overwrite
previsoes_para_salvar.write.csv("hdfs:///opt/spark/data/previsoesnovosdados", mode="overwrite")

# Encerrando a Spark Session
spark.stop()
