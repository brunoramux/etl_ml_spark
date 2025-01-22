# Projeto 5 - Otimização de Pipeline ETL e Machine Learning com PySpark
# Módulo Principal do Pipeline 

# Imports
import os
import traceback
import pyspark 
from pyspark.sql import SparkSession
from p5_log import dsa_grava_log
from p5_processamento import dsa_limpa_transforma_dados
from p5_ml import dsa_cria_modelos_ml

# Cria a sessão e grava o log no caso de erro
try:
	spark = SparkSession.builder.appName("DSAProjeto5").getOrCreate()
	spark.sparkContext.setLogLevel("ERROR")
except:
	dsa_grava_log("Log DSA - Ocorreu uma falha na Inicialização do Spark.")
	dsa_grava_log(traceback.format_exc())
	raise Exception(traceback.format_exc())

# Grava o log
dsa_grava_log("\nLog DSA - Iniciando o Projeto 5.")
dsa_grava_log("Log DSA - Spark Inicializado.")

# Bloco de limpeza e transformação
try:
	DadosHTFfeaturized, DadosTFIDFfeaturized, DadosW2Vfeaturized = dsa_limpa_transforma_dados(spark)
except:
	dsa_grava_log("Log DSA - Ocorreu uma falha na limpeza e transformação dos dados.")
	dsa_grava_log(traceback.format_exc())
	spark.stop()
	raise Exception(traceback.format_exc())

# Bloco de criação dos modelos de Machine Learning
try:
	dsa_cria_modelos_ml(spark, DadosHTFfeaturized, DadosTFIDFfeaturized, DadosW2Vfeaturized)
except:
	dsa_grava_log("Log DSA - Ocorreu Alguma Falha ao Criar os Modelos de Machine Learning.")
	dsa_grava_log(traceback.format_exc())
	spark.stop()
	raise Exception(traceback.format_exc())

# Grava o log
dsa_grava_log("Log DSA - Modelos Criados e Salvos com Sucesso.")

# Grava o log
dsa_grava_log("Log DSA - Processamento Finalizado com Sucesso.\n")

# Finaliza o Spark (encerra o cluster)
spark.stop()



