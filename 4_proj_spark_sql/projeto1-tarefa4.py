# Projeto 1 - Pipelines de Exploração de Dados e Operações SQL com Spark SQL
# Tarefa 4

# Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, trim

# Cria uma sessão Spark
spark = SparkSession.builder \
    .appName("Pipeline de Exploração de Dados com Spark SQL - Tarefa 4") \
    .getOrCreate()

# Carrega o dataset
df_dsa = spark.read.text("data/dados2_cap03.txt")

# Considerando que cada linha tenha o formato 'id, nome, idade'
df_clean = df_dsa.select(
    split(col("value"), ",")[0].alias("id"),
    split(col("value"), ",")[1].alias("nome"),
    split(col("value"), ",")[2].alias("idade")
)

# Removendo espaços em branco e convertendo idade para inteiro
df_clean = df_clean.withColumn("nome", trim(col("nome")))\
    .withColumn("idade", col("idade").cast("integer"))

# Cria tabela temporária
df_clean.createOrReplaceTempView("pessoas")

# Consulta SQL
resultado = spark.sql("SELECT * FROM pessoas WHERE idade > 30")

# Mostra o resultado
resultado.show()

# Salva o resultado em um arquivo CSV
resultado.write.mode('overwrite').csv('data/resultado_tarefa4', header = True)

# Encerra a sessão Spark
spark.stop()
