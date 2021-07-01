from Modulos import EntradaUsuario
from Modulos.ManipArquivosDataset import CarregarDataset, NormalizarDataset
from Modulos.TreinamentoRedeNeural import IniciaTreinoTeste

def Start():
    # le os dados convertidos da linha de comando.
    dados_convertidos = EntradaUsuario.TratamentoLinhaComando(None)
    
    # se nao entrar o -n, eu emito erro e encerro a aplicacao.
    if dados_convertidos["n"][0] not in [0, 1]:
        print("Erro: parametro -n não definido.")
        return
    else:
        # carrega a base de treino e de testes.
        dataset_treino, dataset_testes = CarregarDataset("Datasets/emnist-digits-train.csv", "Datasets/emnist-digits-test.csv")
           
        # formatando os dados de entrada.
        numero_saidas = 1   
        
        # quebro a base de treino e de teste em entradas e saidas.
        entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste = NormalizarDataset(dataset_treino, dataset_testes, numero_saidas, dados_convertidos["n"])
          
        # inicia o treinamento.
        IniciaTreinoTeste(entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste, dados_convertidos["fa"], dados_convertidos["m"], dados_convertidos["ta"], dados_convertidos["t"], dados_convertidos["bs"], dados_convertidos["nc"], dados_convertidos["arq"][0])

    
    
def MainP2(entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste, string_entrada):
    # inicio a execucao do treinamento, com os parametros passados via string de entrada.
    EntradaUsuario.Start2(entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste, string_entrada)
    
if __name__ == "__main__":
    # se este arquivo for o main, inicio a execucao do treinamento, com os parametros passados por argv.
    Start()