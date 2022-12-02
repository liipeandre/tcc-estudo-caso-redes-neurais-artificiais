from modules.dataset import *
from modules.paralelism import *
from modules.file_manipulation import print_format, write_neural_network, write_on_output
from modules.neural_network import create_neural_network, train_neural_network, test_neural_network

from os import cpu_count
from json import loads


def run(parameters, dataset):
    print(print_format(parameters, "Iniciado"))
    print(print_format(parameters, "Criando e treinando a rede neural artificial..."))

    neural_network = create_neural_network(parameters)

    neural_network, status = train_neural_network(
        neural_network,
        dataset["entrada_treino_normalizada"] if parameters["normalizar_dataset"] else dataset["entrada_treino"],
        dataset["saida_treino"]
    )

    print(print_format(parameters, "Processo concluido."))
    print(print_format(parameters, "Testando rede neural artificial..."))

    score = test_neural_network(
        neural_network,
        dataset["entrada_teste_normalizada"] if parameters["normalizar_dataset"] else dataset["entrada_teste"],
        dataset["saida_teste"]
    )

    print(print_format(parameters, "Processo concluido."))
    print(print_format(parameters, "Salvando rede neural artificial e os resultados obtidos..."))

    write_neural_network(neural_network, parameters)
    write_on_output(parameters, score, status)

    print(print_format(parameters, "Processo concluido."))


def main():
    print("Carregando dataset...")

    dataset_treino = load_dataset("datasets/emnist-digits-train.csv")
    dataset_teste = load_dataset("datasets/emnist-digits-test.csv")

    print("Processo concluido.")
    print("Separando os parametros e normalizando-os...")

    entrada_treino, saida_treino = split_dataset(dataset_treino)
    entrada_teste, saida_teste = split_dataset(dataset_teste)

    entrada_treino_normalizada = normalize_dataset(entrada_treino)
    entrada_teste_normalizada = normalize_dataset(entrada_teste)

    dataset = {
        "entrada_treino": entrada_treino,
        "entrada_treino_normalizada": entrada_treino_normalizada,
        "saida_treino": saida_treino,
        "entrada_teste": entrada_teste,
        "entrada_teste_normalizada": entrada_teste_normalizada,
        "saida_teste": saida_teste,
    }

    print("Processo concluido.")
    print("Criando a fila de threads de treinamento...")

    test_cases = load_test_cases()

    threads = []
    for test_case in test_cases:
        thread = create_thread(loads(test_case), dataset, run)
        threads.append(thread)

    print("Processo concluido.")
    print("Iniciando treinamento...")

    while threads:
        max_threads = cpu_count()
        for thread in threads[:max_threads]:
            if thread.ident is None:
                thread.start()

        for thread in threads[:max_threads]:
            if not thread.is_alive():
                thread.join()
                threads.remove(thread)


if __name__ == '__main__':
    main()
