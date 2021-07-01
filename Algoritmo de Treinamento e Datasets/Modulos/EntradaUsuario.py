from Modulos.Bibliotecas import *
from Modulos.TreinamentoRedeNeural import IniciaTreinoTeste
from Modulos.ManipArquivosDataset import CarregarDataset, NormalizarDataset

def Start2(entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste, string_entrada):
    # le e converte os dados da linha de comando ou de uma string de entrada.
    dados_convertidos = TratamentoLinhaComando(string_entrada)
      
    # inicia o treinamento.
    IniciaTreinoTeste(entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste, dados_convertidos["fa"], dados_convertidos["m"], dados_convertidos["ta"], dados_convertidos["t"], dados_convertidos["bs"], dados_convertidos["nc"], dados_convertidos["arq"][0])

    
def Gerador(repetir, lista, inicio_interv, fim_interv):  
    # gero o produto cartesiano de topologias com os valores da lista acima.
    # usado no teste do número de camadas ocultas.
    lista_topologias = list(product(lista, repeat=repetir))
    
    # converto de lista para string apenas uma parte da lista gerada.
    string_entrada_aux = []
    for item in lista_topologias[inicio_interv:fim_interv]:
        string_entrada_aux.append(str(list(item)))
    
    # concatenando todas as substrings em uma unica string de saida, além de removendo o espaço em branco entre o [].
    string_entrada = ""
    for x in string_entrada_aux:
        string_entrada += x.replace(" ", "") + " "
        
    # retorno a string gerada para ser usada pelo processo e o número de topologias geradas.
    return string_entrada
    
    
def TratamentoLinhaComando(string_entrada):
    
    # antes de processar a string de entrada, faco a quebra da string.
    if string_entrada is not None:
        string_entrada = string_entrada.split()
   
    # definindo a formatacao da string de entrada.
    parser = ArgumentParser(formatter_class=RawTextHelpFormatter)
    
    # -n   --> indica se irá ser feita normalizacao da base de treino.
    # -fa  --> lista de funcoes de ativacao.
    # -m   --> lista de momentum a serem avaliados.
    # -ta  --> lista de taxas de aprendizado a serem avaliados
    # -t   --> topologia das camadas intermediarias (quantidade e numero de neuronios em cada uma delas). Representado por uma lista de listas ("matriz").
    # -bs  --> tamanho do bloco usado em cada ciclo (-bs = 1 --> treinamento on-line)
    # -nc  --> número máximo de ciclos.
    # -arq --> nome do arquivo de saída.
    
    parser.add_argument("-n", nargs=1, type=int, choices=[0, 1], required=False, help=dedent(
    '''\t\t\t\t\tIndica se a normalizacao ira ser feita.\n\n\n\n\n'''))
    
    parser.add_argument("-fa", nargs="+", choices=["tanh", "logistic"], required=True, help=dedent(
    '''\t\t\t\t\tLista com as funcoes de ativacao a serem testadas.\n\n\n\n\n'''))

    parser.add_argument("-m", nargs="+", type=float, required=True, help=dedent(
    '''Lista com os valores para momentum a serem testados.\n\n\n\n\n'''))

    parser.add_argument("-ta", nargs="+", type=float, required=True, help=dedent(
    '''Lista com os valores para taxa de aprendizado a serem testadas.\n\n\n\n\n'''))

    parser.add_argument("-t", nargs="+", required=True, help=dedent(
    '''Listas com as topologias de rede a serem testadas.

    Formato de cada entrada: [ci_1, ci_2, ci_3, ..., ci_n].
    Onde ci_n sao camadas intermediarias da rede neural. Entre ci_n e ci_n+1 nao deve existir espaços\n\n\n\n\n'''))

    parser.add_argument("-bs", nargs="+", type=int, required=True, help=dedent(
    '''\t\t\t\t\tLista com os tamanhos de bloco a serem testados.\n\n\n\n\n'''))
    
    parser.add_argument("-nc", nargs="+", type=int, required=True, help=dedent(
    '''\t\t\t\t\tLista com os numeros maximos de ciclos a serem testados.\n\n\n\n\n'''))
    
    parser.add_argument("-arq", nargs=1, required=True, help=dedent(
    '''Nome do arquivo de saida com os resultados dos testes.\n\n\n\n\n'''))

    # faço a leitura dos parametros de entrada e converto os dados de entrada.
    # parser faz o pre-tratamento dos dados de entrada e conversao dos dados de entrada de string para list.
    # se for string não for None, trate a própria string. 
    if string_entrada is not None:
        dados_convertidos = vars(parser.parse_args(string_entrada))
    else:
    # se for string vazia, trate o argv.
        dados_convertidos = vars(parser.parse_args())
    
    # converto de string para lista de listas a string de topologias.
    dados_convertidos_t_aux = []
    for elemento in dados_convertidos["t"]:
        dados_convertidos_t_aux.append([x for x in map(uint16, elemento.strip('[]').split(','))])
    
    # armazeno os dados do auxiliar para a variavel original.
    dados_convertidos["t"] = dados_convertidos_t_aux
    
    # retorno os dados convertidos
    return dados_convertidos