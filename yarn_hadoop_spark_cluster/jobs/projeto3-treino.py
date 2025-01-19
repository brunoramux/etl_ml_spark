# Projeto 3 - Pipeline de Machine Learning em Cluster Spark Para Previsão de Churn - Treinamento e Deploy
# Script de Treino do Modelo

# Imports
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Inicializando a Spark Session
spark = SparkSession.builder.appName("Projeto3-Treino").getOrCreate()

# Leitura dos Dados do HDFS (coloque o arquivo no HDFS como mostrado nas aulas)
df_dsa = spark.read.csv("hdfs:///opt/spark/data/dataset.csv", header=True, inferSchema=True)

# Processamento dos Dados

# Convertendo colunas categóricas para representação numérica
indexers = [StringIndexer(inputCol=column, outputCol=column+"_index").fit(df_dsa) for column in ["Plano", "TempoContrato"]]

# Criando o vetor de recursos
assembler = VectorAssembler(inputCols=["Idade", 
                                       "UsoMensal", 
                                       "SatisfacaoCliente", 
                                       "ValorMensal", 
                                       "Plano_index", 
                                       "TempoContrato_index"], 
                            outputCol="features")

# Construção do Modelo de Machine Learning

# Dividindo os dados em conjunto de treino e teste
dados_treino, dados_teste = df_dsa.randomSplit([0.7, 0.3])

# Inicializando o modelo definindo dados de entrada e saída
modelo_rf = RandomForestClassifier(labelCol="Churn", featuresCol="features")

# Criando o pipeline
pipeline = Pipeline(stages=indexers + [assembler, modelo_rf])

# Treinando o modelo
modelo_dsa = pipeline.fit(dados_treino)

# Previsões com dados de teste
previsoes = modelo_dsa.transform(dados_teste)

# Avaliando o modelo
avaliador = BinaryClassificationEvaluator(labelCol="Churn")
acuracia = avaliador.evaluate(previsoes)

# Criando um DataFrame com a acurácia (Vamos salvar no HDFS no formato CSV)
acuracia_df = spark.createDataFrame([Row(acuracia=acuracia)])

# Selecionando as colunas necessárias para salvar no HDFS
selected_columns = ["Idade", "UsoMensal", "Plano", "SatisfacaoCliente", "TempoContrato", "ValorMensal", "Churn", "prediction"]
previsoes_para_salvar = previsoes.select(selected_columns)

# Salvando no HDFS
modelo_dsa.write().overwrite().save("hdfs:///opt/spark/data/modelo")
acuracia_df.write.csv("hdfs:///opt/spark/data/acuracia", mode="overwrite")
previsoes_para_salvar.write.csv("hdfs:///opt/spark/data/previsoes", mode="overwrite")

# Encerrando a Spark Session
spark.stop()
