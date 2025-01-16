# Projeto 1 - Pipelines de Exploração de Dados e Operações SQL com Spark SQL
# Tarefa 1

# Imports
from pyspark.sql import SparkSession

# Cria uma sessão Spark
spark = SparkSession.builder \
    .appName("Pipeline de Exploração de Dados com Spark SQL - Tarefa 1") \
    .getOrCreate()

# Configura o nível de log
spark.sparkContext.setLogLevel('ERROR')

# Carrega o conjunto de dados no formato CSV
df_dsa = spark.read.csv('/opt/spark/data/dados1_cap03.csv', header = True, inferSchema = True)

# Mostra as primeiras linhas do DataFrame
df_dsa.show()

# Realiza algumas operações de exploração
print(f"Total de registros: {df_dsa.count()}")
print("Distribuição de registros por ano:")
df_dsa.groupBy("ano").count().show()

# Cria uma view temporária para executar consultas SQL
df_dsa.createOrReplaceTempView("vendas")

# Executa uma consulta SQL
resultado = spark.sql("""
    SELECT ano, ROUND(AVG(unidades_vendidas), 2) as media_unidades_vendidas
    FROM vendas
    GROUP BY ano
    ORDER BY media_unidades_vendidas DESC
""")
    
# Mostra o resultado
resultado.show()

# Salvar o resultado em um arquivo CSV
resultado.write.csv('data/resultado_tarefa1', header = True)

# Encerrar a sessão Spark
spark.stop()
