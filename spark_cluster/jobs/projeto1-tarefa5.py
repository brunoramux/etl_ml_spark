# Projeto 1 - Pipelines de Exploração de Dados e Operações SQL com Spark SQL
# Tarefa 5

# Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, trim, when

# Cria uma sessão Spark
spark = SparkSession.builder \
    .appName("Pipeline de Exploração de Dados com Spark SQL - Tarefa 5") \
    .getOrCreate()

# Carrega o dataset
df_dsa = spark.read.text("data/dados2_cap03.txt")

# Considerando que cada linha tenha o formato 'id, nome, idade'
df_clean = df_dsa.select(
    split(col("value"), ",")[0].alias("id"),
    split(col("value"), ",")[1].alias("nome"),
    split(col("value"), ",")[2].alias("idade")
)

# Removendo espaços em branco, convertendo idade para inteiro e adicionando a coluna faixa_etaria
df_clean = df_clean.withColumn("nome", trim(col("nome")))\
    .withColumn("idade", col("idade").cast("integer"))\
    .withColumn("faixa_etaria", 
                when(col("idade") < 18, "Jovem")
                .when((col("idade") >= 18) & (col("idade") <= 60), "Adulto")
                .otherwise("Idoso"))

# Cria tabela temporária
df_clean.createOrReplaceTempView("pessoas")

# Consulta SQL
resultado = spark.sql("SELECT id, nome, faixa_etaria FROM pessoas WHERE faixa_etaria = 'Adulto' OR faixa_etaria = 'Idoso'")

# Mostra o resultado
resultado.show()

# Salva o resultado em um arquivo CSV
resultado.write.mode('overwrite').csv('data/resultado_tarefa5', header = True)

# Encerra a sessão Spark
spark.stop()
