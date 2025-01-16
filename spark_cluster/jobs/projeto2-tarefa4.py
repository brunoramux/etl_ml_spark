# Projeto 2 - Banco de Dados, Machine Learning e Pipeline ETL em Cluster Spark Para Detectar Anomalias em Transações Financeiras
# Tarefa 4 - Criando o Quarto Pipeline Para Transformação de Dados com Machine Learning

# Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import year, month, dayofmonth, hour, count
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans

# Inicializando a sessão Spark
spark = SparkSession.builder \
    .appName("DSA Projeto 2 - Tarefa 4") \
    .getOrCreate()

# Configurando o nível de log
spark.sparkContext.setLogLevel('ERROR')

# Carregando dados
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

# Normalizando os dados (requisito para o algoritmo)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures")
df_dsa = scaler.fit(df_dsa).transform(df_dsa)

# Cria o modelo KMeans
kmeans = KMeans(featuresCol='scaledFeatures', k=3)  

# Treinar o modelo KMeans
modelo = kmeans.fit(df_dsa)

# Atribuindo pontos aos clusters
df_resultado = modelo.transform(df_dsa)

# Remove a coluna 'features' e 'scaledFeatures' pois não são mais necessárias
df_resultado_clean = df_resultado.drop('features', 'scaledFeatures')
df_resultado_clean.show()

# Imprimindo as primeiras linhas do DataFrame
df_resultado_clean.show()

# Contagem total de linhas no DataFrame 
print(f"Total de linhas no DataFrame: {df_resultado_clean.count()}")

df_resultado.groupBy('prediction').count().show()

df_resultado.groupBy('prediction').agg(count('Id_transacao').alias('Qtde')).show()

# Salvando o resultado em um arquivo CSV
df_resultado_clean.write.mode('overwrite').csv('data/resultado_tarefa4', header = True)

# Encerrando a sessão Spark
spark.stop()

print("\nObrigado DSA. Execução do Job Concluída com Sucesso!\n")

