# Projeto 2 - Banco de Dados, Machine Learning e Pipeline ETL em Cluster Spark Para Detectar Anomalias em Transações Financeiras
# Tarefa 1 - Criando o Primeiro Pipeline Para Listar Anomalias nas Transações Financeiras

# Imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Inicializando a sessão Spark
spark = SparkSession.builder \
    .appName("DSA Projeto 2 - Tarefa 1") \
    .getOrCreate()

# Carregando dados de um arquivo CSV
df_dsa = spark.read.csv('data/dados1_cap05.csv', header = True, inferSchema = True)

# Configurando o nível de log
spark.sparkContext.setLogLevel('ERROR')

# Imprimindo os dados 
df_dsa.show()

# Definindo um limite para considerar uma transação como anomalia
LIMITE_ANOMALIA = 50000 

# Filtrando transações que são consideradas anomalias
anomalias = df_dsa.filter(col('Valor') > LIMITE_ANOMALIA)

# Imprimindo anomalias detectadas
anomalias.show()

# Contagem total de linhas no DataFrame de anomalias
print(f"Total de linhas no DataFrame de anomalias: {anomalias.count()}")

# Salvando o resultado em um arquivo CSV 
anomalias.write.mode('overwrite').csv('data/resultado_tarefa1', header = True)

# Encerrando a sessão Spark
spark.stop()

print("\nObrigado DSA. Execução do Job Concluída com Sucesso!\n")


