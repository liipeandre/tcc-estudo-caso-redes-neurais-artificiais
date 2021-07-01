from Modulos.Bibliotecas import *

def CarregarDataset(diretorio_dataset_treino, diretorio_dataset_teste):
    system("cls")
    print("Carregando Datasets")

    # uma lista em python e equivalente a um vetor em C/C++, porem podem receber qualquer tipo de dado (assim como qualquer variavel em python).
    # dataset_treino e dataset_teste são listas de listas (equivalente a uma matriz no C/C++).

    # carrego os datasets para treino e testes em arquivos separados.
    dataset_treino = loadtxt(diretorio_dataset_treino, dtype=uint8, delimiter=";")
    dataset_teste = loadtxt(diretorio_dataset_teste, dtype=uint8, delimiter=";")

    print("Carregamento concluido\n\n")
    
    # retorno uma tupla com o conteúdo das duas variaveis.
    # uma tupla é equivalente a uma lista, porém imutável (não permite inserção de novos elementos, mas permite acesso aos dados dele) e usado frequentemente em retorno de função.
    return (dataset_treino, dataset_teste)

    
    
    
def GravarResultadosTeste(string_formatada, arquivo_saida):
    # gravo cada elemento da lista numa linha do arquivo.
    arquivo_saida.write(str(string_formatada) + "\n")
    return

    
    
    
def GravarRedeNeuralTreinada(rede_neural, diretorio_salvamento):   
    # salva a rede treinada no arquivo.
    # esse arquivo esta no formato binario.
    dump(rede_neural, open(diretorio_salvamento+ ".rna", "wb"))
    return

    
    
    
def NormalizarDataset(dataset_treino, dataset_testes, numero_saidas, normalizar):
    # sintaxe python para manipulacao de partes de uma lista (list slicing):

    # x:y --> intervalo desde o elemento na posicao x da lista (contando a partir do 0) ate o elemento na posicao y-1 da lista (contando a partir do 0 e considerando intervalo aberto).
    # x:  --> intervalo desde o elemento na posicao x da lista (contando a partir do 0) ate o fim da lista.
    # :x  --> intervalo desde o comeco da lista até o elemento na posicao x-1 da lista (contando a partir do 0 e considerando intervalo aberto).
    # :   --> intervalo desde o comeco da lista até o fim da lista (ou seja, toda a lista).    

    # separo em características de entrada e saida esperada (classe) os datasets de treino e teste.
    entrada_treino = dataset_treino[:, numero_saidas:]
    saida_esperada_treino = dataset_treino[:, :numero_saidas]

    entrada_teste = dataset_testes[:, numero_saidas:]
    saida_esperada_teste = dataset_testes[:, :numero_saidas]
   
   # se normalizar for True, normaliza a base de treino e teste para melhorar o treinamento.
    if(normalizar == True):
        normalizador = StandardScaler()
        normalizador.fit(entrada_treino)    
        
        entrada_treino = normalizador.transform(entrada_treino).astype(float16)
        entrada_teste = normalizador.transform(entrada_teste).astype(float16)
   
    # retorno uma tupla com o conteúdo das quatro variaveis.
    # uma tupla é equivalente a uma lista, porém imutável (não permite insercao e remocao de novos elementos, mas permite acesso aos dados dele) e usado frequentemente em retorno de função, pois permite o retorno multiplo de variaveis.
    return (entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste)

    
    
    
def FormatarSaidaTXT(funcao_ativacao_atual, topologia_atual, taxa_aprendizado_atual, momentum_atual, batch_atual, maxciclos_atual, resultado):
    # concatenando os dados em uma string de saida (serializacao).
    string_formatada = ("Rede Neural de Funcao de Ativacao = " + str(funcao_ativacao_atual) + 
                        ", Topologia = " + str(topologia_atual) + 
                        ", Taxa de Aprendizado = " + str(taxa_aprendizado_atual) +
                        ", Momentum = " + str(momentum_atual) +
						", Batch Size = " + str(batch_atual) +
						", Num Ciclos = " + str(maxciclos_atual) +
                        ": Precisao (em %) = " + str(resultado))
                        
    # retorno a string com os dados a serem escritos no arquivo saida.
    return string_formatada

    
    
    
def FormatarNomeArquivo(diretorio_salvamento, topologia_atual, funcao_ativacao_atual, taxa_aprendizado_atual, momentum_atual, batch_atual, maxciclos_atual):
    # concatenando os dados em uma string de saida (serializacao).
    nome_arquivo_saida = (str(diretorio_salvamento) + 
                          str(funcao_ativacao_atual) + "-" + 
                          str(taxa_aprendizado_atual) + "-" + 
                          str(momentum_atual) + "-" + 
                          str(topologia_atual) + "-" + 
						  str(batch_atual) + "-" +
						  str(maxciclos_atual) + ".rna")
    
    # retorno a string com os dados a serem escritos no arquivo saida.
    return nome_arquivo_saida