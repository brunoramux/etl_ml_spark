    # Projeto 1 - Pipelines de Exploração de Dados e Operações SQL com Spark SQL
    # Tarefa 2

    # Imports
    from pyspark.sql import SparkSession

    # Cria uma sessão Spark
    spark = SparkSession.builder \
        .appName("Pipeline de Exploração de Dados com Spark SQL - Tarefa 2") \
        .getOrCreate()

    # Configura o nível de log
    spark.sparkContext.setLogLevel('ERROR')

    # Carrega o conjunto de dados CSV
    df_dsa = spark.read.csv('data/dados1_cap03.csv', header = True, inferSchema = True)

    # Cria uma view temporária para executar consultas SQL
    df_dsa.createOrReplaceTempView("vendas")

    # Executa uma consulta SQL
    resultado = spark.sql("""
        SELECT ano, funcionario, SUM(unidades_vendidas) AS total_unidades_vendidas
        FROM vendas
        GROUP BY ano, funcionario
        ORDER BY ano, funcionario
    """)

    # Salva o resultado em um arquivo CSV
    resultado.write.csv('data/resultado_tarefa2', header = True)

    # Convertendo o Spark DataFrame para Pandas DataFrame
    resultado_pd = resultado.toPandas()

    # Importando Matplotlib
    import matplotlib.pyplot as plt

    # Criando um gráfico de barras
    plt.figure(figsize=(10,6))
    plt.bar(resultado_pd['funcionario'], resultado_pd['total_unidades_vendidas'])
    plt.xlabel('Funcionário')
    plt.ylabel('Total de Unidades Vendidas')
    plt.title('Vendas por Funcionário por Ano')

    # Salvando o gráfico em um arquivo
    plt.savefig('data/resultado_tarefa2/tarefa2.png')

    # Encerrar a sessão Spark
    spark.stop()
