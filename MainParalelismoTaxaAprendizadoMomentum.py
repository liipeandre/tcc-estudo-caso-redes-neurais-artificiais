from Modulos.ManipArquivosDataset import CarregarDataset, NormalizarDataset
from Modulos.Bibliotecas import *
from MainTreinamento import MainP2

def MainP1():
    # declaro uma string vazia.
    normalizar = ""
    
    # enquanto normalizar nao for sim ou nao (maiusculo)
    while normalizar not in ["SIM", "NAO"]:
        
        # limpo a tela, imprimo a mensagem e leio do teclado.
        system("cls")
        normalizar = input("Normalizar? (Sim/Nao): ")
        
        # converto para maiusculo, para tratar apenas dois casos.
        normalizar = normalizar.upper()

    # carrega a base de treino e de testes.
    dataset_treino, dataset_testes = CarregarDataset("Datasets/emnist-digits-train.csv", "Datasets/emnist-digits-test.csv")
   
    # formatando os dados de entrada.
    numero_saidas = 1     
    
    # quebro a base de treino e de teste em entradas e saidas, normalizo se a variavel "normalizar" for "SIM".
    entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste = NormalizarDataset(dataset_treino, dataset_testes, numero_saidas, True if normalizar=="SIM" else False)
        
    # crio a lista com os parâmetros para cada thread.
    lista_strings_entrada = ["-fa tanh logistic -m 0.0 -ta 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -t [10] [25] [50] [100] [200] -bs 234 -nc 1 -arq ResultadoTestesParte1.txt",
                             "-fa tanh logistic -m 0.0 -ta 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -t [300] [400] [500] [1000] [2000] -bs 234 -nc 1 -arq ResultadoTestesParte2.txt",
                             "-fa tanh logistic -m 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -ta 0.1 -t [10] [25] [50] [100] [200] -bs 234 -nc 1 -arq ResultadoTestesParte3.txt",
                             "-fa tanh logistic -m 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 -ta 0.1 -t [300] [400] [500] [1000] [2000] -bs 234 -nc 1 -arq ResultadoTestesParte4.txt"]
    
    # para cada string de entrada, eu crio uma thread e a inicio.
    lista_threads = []
    for string_entrada in lista_strings_entrada:
        
        # adiciono a thread na lista de threads.
        lista_threads.append(Thread(target=MainP2, args=(entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste, string_entrada)))
        
        # inicio a última thread criada (-1 é o fim da lista, ou seja, o elemento recém inserido).
        lista_threads[-1].start()
    
    # aguardo todas as threads terminarem.
    for thread in lista_threads:
        thread.join()
 
    # desliga o computador depois de terminado.
    #system("shutdown -s")
    
if __name__ == "__main__":
    MainP1()