from Modulos.Bibliotecas import *
from Modulos.ManipArquivosDataset import *

def IniciaTreinoTeste(entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste, lista_funcoes_ativacao, lista_momentum, lista_taxa_aprendizado, lista_topologias, lista_batch, lista_maxciclos, nome_arquivo_saida):
    # abrindo o arquivo que armazenará os resultados.
    arquivo_saida = open(nome_arquivo_saida, "w")

    # iniciando o treinamento e testes da rede neural artificial.
    TreinoTeste(arquivo_saida, lista_funcoes_ativacao, lista_topologias, lista_momentum, lista_taxa_aprendizado, lista_batch, lista_maxciclos, entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste)

    # fecho o arquivo de resultados.
    arquivo_saida.close()


def TreinoTeste(arquivo_saida, lista_funcoes_ativacao, lista_topologias, lista_momentum, lista_taxa_aprendizado, lista_batch, lista_maxciclos, entrada_treino, saida_esperada_treino, entrada_teste, saida_esperada_teste):
    # quando se utiliza o operador "in" em um for, ele é semelhante a um for_each() em C/C++/Java/C#, etc...
    for funcao_ativacao_atual in lista_funcoes_ativacao:
        for topologia_atual in lista_topologias:
            for momentum_atual in lista_momentum:
                for taxa_aprendizado_atual in lista_taxa_aprendizado:
                    for batch_atual in lista_batch:
                        for maxciclos_atual in lista_maxciclos:
                    
                            # exibe dados do treinamento atual.
                            print("Treinando a rede neural artificial de: ")
                            print("Funcao de ativacao: " + str(funcao_ativacao_atual))
                            print("Topologia: " + str(topologia_atual))
                            print("Taxa de aprendizado: " + str(taxa_aprendizado_atual))
                            print("Momentum: " + str(momentum_atual) + "\n")
                            print("Batch Size: " + str(batch_atual))
                            print("Numero Ciclos: " + str(maxciclos_atual) + "\n")

                            # define a variável que sera a rede neural.
                            rede_neural = MLPClassifier(hidden_layer_sizes=topologia_atual, activation=funcao_ativacao_atual, solver="sgd", learning_rate="constant", learning_rate_init=taxa_aprendizado_atual, momentum=momentum_atual, max_iter=maxciclos_atual, shuffle=False, batch_size=batch_atual)      
                            
                            # faco o treinamento da rede neural.
                            # ravel() converte a lista ("vetor") de lista coluna ("vetor" coluna) para lista linha ("vetor" linha).
                            # se a rede neural nao convergir, a bilbioteca lança uma exceção, logo devo capturá-la.
                            try:
                                rede_neural.fit(entrada_treino, saida_esperada_treino.ravel())
                            except ConvergenceWarning:
                                
                                # crio uma trava de arquivo (para impedir que duas ou mais threads escrevam no arquivo ao mesmo tempo).
                                lock = FileLock("ProtecaoErrosTreinamentoTxt.lock")
                                
                                # se a trava permitir o acesso (se outra thread estiver acessando o arquivo, essa thread vai "dormir" até o status da trava mudar para "liberado").
                                with lock:
                                
                                    # abro/crio o arquivo.
                                    arquivo = open("ErrosTreinamento.txt", "a")
                                    
                                    # escrevo o erro nele.
                                    arquivo.write("Rede Neural de Funcao de Ativacao = " + str(funcao_ativacao_atual) + 
                                                ", Topologia = " + str(topologia_atual) + 
                                                ", Taxa de Aprendizado = " + str(taxa_aprendizado_atual) +
                                                ", Momentum = " + str(momentum_atual) +
                                                ", Batch Size = " + str(batch_atual) +
                                                ", Numero Ciclos = " + str(maxciclos_atual) +
                                                ": Nao Convergiu")
                                    
                                    # fecho o arquivo.
                                    arquivo.close()
                                    
                                # aqui, a trava está desbloqueada (fim da regiao critica). 
                            
                            # avalia o desempenho da rede neural.
                            resultado = rede_neural.score(entrada_teste, saida_esperada_teste.ravel()) * 100

                            # define o formato de saida do resultado.
                            string_formatada = FormatarSaidaTXT(funcao_ativacao_atual, topologia_atual, taxa_aprendizado_atual, momentum_atual, batch_atual, maxciclos_atual, resultado)                
                            nome_arquivo_rede_neural = FormatarNomeArquivo("RedesTreinadas/rede_treinada", topologia_atual, funcao_ativacao_atual, taxa_aprendizado_atual, momentum_atual, batch_atual, maxciclos_atual)
                            
                            # salva a rede neural e o resultado no arquivo.
                            GravarRedeNeuralTreinada(rede_neural, nome_arquivo_rede_neural)
                            GravarResultadosTeste(string_formatada, arquivo_saida)
                            print("Procedimento concluido!")
                            system("cls")