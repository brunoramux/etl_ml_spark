# Projeto 1 - Pipelines de Exploração de Dados e Operações SQL com Spark SQL
# Tarefa 3

# Imports
from pyspark.sql import SparkSession

# Cria uma sessão Spark
spark = SparkSession.builder \
    .appName("Pipeline de Exploração de Dados com Spark SQL - Tarefa 3") \
    .getOrCreate()

# Configura o nível de log
spark.sparkContext.setLogLevel('ERROR')

# Carrega o conjunto de dados CSV
df_dsa = spark.read.csv('data/dados1_cap03.csv', header = True, inferSchema = True)

# Cria uma view temporária para executar consultas SQL
df_dsa.createOrReplaceTempView("vendas")

# Executa uma consulta SQL
resultado = spark.sql("""
    WITH vendas_agregadas AS (
    SELECT ano, funcionario, SUM(unidades_vendidas) AS total_unidades_vendidas
    FROM vendas
    GROUP BY ano, funcionario
),
total_ano AS (
    SELECT ano, SUM(total_unidades_vendidas) AS total_unidades_ano
    FROM vendas_agregadas
    GROUP BY ano
)
SELECT v.ano,
       v.funcionario,
       v.total_unidades_vendidas,
       t.total_unidades_ano,
       ROUND(v.total_unidades_vendidas / t.total_unidades_ano * 100, 2) AS proporcional_func_ano
FROM vendas_agregadas v
JOIN total_ano t ON v.ano = t.ano
ORDER BY v.ano, v.funcionario;
""")

# Mostra o resultado
resultado.show()

# Salva o resultado em um arquivo CSV
resultado.write.mode('overwrite').csv('data/resultado_tarefa3', header = True)

# Executa a consulta SQL com EXPLAIN para visualizar o plano de execução da query
resultado_explain = spark.sql("""
    WITH vendas_agregadas AS (
    SELECT ano, funcionario, SUM(unidades_vendidas) AS total_unidades_vendidas
    FROM vendas
    GROUP BY ano, funcionario
),
total_ano AS (
    SELECT ano, SUM(total_unidades_vendidas) AS total_unidades_ano
    FROM vendas_agregadas
    GROUP BY ano
)
SELECT v.ano,
       v.funcionario,
       v.total_unidades_vendidas,
       t.total_unidades_ano,
       ROUND(v.total_unidades_vendidas / t.total_unidades_ano * 100, 2) AS proporcional_func_ano
FROM vendas_agregadas v
JOIN total_ano t ON v.ano = t.ano
ORDER BY v.ano, v.funcionario;
""")

# Mais imports
import io
import sys

# Redireciona a saída padrão (stdout) para uma variável
old_stdout = sys.stdout
new_stdout = io.StringIO()
sys.stdout = new_stdout

# Gera o plano de execução
resultado_explain.explain()

# Restaura a saída padrão e obtém o conteúdo do plano de execução
sys.stdout = old_stdout
plano_execucao = new_stdout.getvalue()

# Escreve o plano de execução em um arquivo
with open("data/resultado_tarefa3/plano_execucao.txt", "w") as file:
    file.write(plano_execucao)

# Encerra a sessão Spark
spark.stop()
