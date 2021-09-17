from math import ceil

from modules.file_manipulation import write_test_cases


def read_test_case_parts():
    return int(input("Digite o número de partes (arquivos) em que os casos de teste serão divididos: \n"))


def generate_test_cases(parameters, parts):
    total_test_cases = len(parameters['normalizar_dataset']) * \
                       len(parameters['funcoes_ativacao']) * \
                       len(parameters['topologias']) * \
                       len(parameters['momentum']) * \
                       len(parameters['taxas_aprendizado']) * \
                       len(parameters['batch_size']) * \
                       len(parameters['max_ciclos'])

    current_part = 1
    test_cases_current_part = 0
    test_cases_per_part = ceil(total_test_cases / parts)

    for normalizar_dataset in parameters['normalizar_dataset']:
        for funcao_ativacao in parameters['funcoes_ativacao']:
            for topologia in parameters['topologias']:
                for momentum in parameters['momentum']:
                    for taxa_aprendizado in parameters['taxas_aprendizado']:
                        for batch_size in parameters['batch_size']:
                            for max_ciclos in parameters['max_ciclos']:

                                test_case = {
                                    "funcao_ativacao": funcao_ativacao,
                                    "topologia": topologia,
                                    "momentum": momentum,
                                    "taxa_aprendizado": taxa_aprendizado,
                                    "batch_size": batch_size,
                                    "max_ciclos": max_ciclos,
                                    "normalizar_dataset": normalizar_dataset
                                }

                                write_test_cases(test_case, current_part)
                                test_cases_current_part += 1

                                if test_cases_current_part == test_cases_per_part:
                                    test_cases_current_part = 0
                                    current_part += 1