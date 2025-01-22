# Projeto 5 - Otimização de Pipeline ETL e Machine Learning com PySpark
# Módulo de Log

# Imports
import os
import os.path
import pendulum
import traceback

# Define a função dsa_grava_log que recebe um texto como parâmetro, que será gravado no log
def dsa_grava_log(texto):
    
    # Define o caminho para armazenar os logs
    path = "/opt/spark/data"
    
    # Obtém o momento atual usando pendulum
    agora = pendulum.now()
    
    # Formata a data atual para o nome do arquivo de log
    data_arquivo = agora.format('YYYYMMDD')
    
    # Formata a data e hora atuais para registrar no log
    data_hora_log = agora.format('YYYY-MM-DD HH:mm:ss')
    
    # Monta o nome do arquivo de log com o caminho, a data e o prefixo -log_spark_p5.txt
    nome_arquivo = path + data_arquivo + "-log_spark_p5.txt"
    
    # Inicializa a variável texto_log como uma string vazia
    texto_log = ''
    
    # Tenta abrir o arquivo de log para escrita, em modo append se o arquivo já existir, ou cria um novo arquivo
    try:

        # Verifica se o arquivo já existe
        if os.path.isfile(nome_arquivo):  

            # Abre o arquivo em modo de adição
            arquivo = open(nome_arquivo, "a")

            # Adiciona uma nova linha se o arquivo já existir  
            texto_log = texto_log + '\n'  

        else:

            # Cria um novo arquivo se ele não existir
            arquivo = open(nome_arquivo, "w")  
    
    # Captura qualquer exceção durante a tentativa de abrir o arquivo
    except:
        print("Erro na tentativa de acessar o arquivo para criar os logs.")

        # Relança a exceção com o traceback para diagnóstico
        raise Exception(traceback.format_exc())  
    
    # Adiciona a data, hora e o texto do log à variável texto_log
    texto_log = texto_log + "[" + data_hora_log + "] - " + texto

    # Escreve o log no arquivo
    arquivo.write(texto_log)  

    # Imprime o texto do log
    print(texto)  

    # Fecha o arquivo após a escrita
    arquivo.close()  
    
