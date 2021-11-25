from json import dumps
from pathlib import Path
from filelock import FileLock
from pickle import dump


def print_format(parameters, status):
    saida = "Topologia: {};\nFuncao Ativacao: {};\nTaxa Aprendizado: {};\nMomentum: {};\nNumero Ciclos: {};\n" \
            "Batch Size: {};\nStatus: {};\n\n\n"

    return saida.format(str(parameters['topologia']),
                        str(parameters['funcao_ativacao']),
                        str(parameters['taxa_aprendizado']),
                        str(parameters['momentum']),
                        str(parameters['max_ciclos']),
                        str(parameters['batch_size']),
                        str(status))



def output_format(parameters, score, status):
    saida = "{};{};{};{};{};{};{};{}\n"

    return saida.format(str(parameters['topologia']),
                        str(parameters['funcao_ativacao']),
                        str(parameters['taxa_aprendizado']),
                        str(parameters['momentum']),
                        str(parameters['max_ciclos']),
                        str(parameters['batch_size']),
                        str(score),
                        str(status))


def filename_format(parameters):
    saida = "{}-{}-{}-{}-{}-{}.rna"

    return saida.format(str(parameters['topologia']),
                        str(parameters['funcao_ativacao']),
                        str(parameters['taxa_aprendizado']),
                        str(parameters['momentum']),
                        str(parameters['max_ciclos']),
                        str(parameters['batch_size']))


def write_on_output(parameters, score, status):
    # Evita que exista concorrência quando duas threads tentarem escrever ao mesmo tempo no arquivo de saída.
    lock = FileLock("output.lock")

    with lock:
        output_dir = "output/score/"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        with open(output_dir + "output.txt", "a") as file:
            output = output_format(parameters, score, status)
            file.write(output)


def write_neural_network(neural_network, parameters):
    output_dir = "output/neural_networks/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    filename = filename_format(parameters)

    with open(output_dir + filename, "wb") as file:
        dump(neural_network, file)


def write_test_cases(test_case, part):
    output_dir = "test_cases/"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with open(output_dir + "{}.txt".format(part), "a") as file:
        file.write(dumps(test_case) + "\n")