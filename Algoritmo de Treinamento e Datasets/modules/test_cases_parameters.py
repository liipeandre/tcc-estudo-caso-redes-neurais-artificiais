from itertools import product


def get_base_parameters():
    return {
        "normalizar_dataset": [True, False],
        "funcoes_ativacao": ['tanh', 'logistic'],
        "topologias": list(product([10, 25, 50, 100, 200, 300, 400, 500, 1000, 2000], repeat=1)),
        "momentum": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "taxas_aprendizado": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "batch_size": [1, 3, 7, 14, 29, 58, 117, 234, 468, 937, 1875, 3750, 7500],
        "max_ciclos": [1, 10, 20, 30, 40, 50, 100, 200, 300, 400, 500, 1000],
    }.copy()


def taxa_aprendizado_test_cases(parameters):
    parameters['momentum'] = [0.0]
    parameters['batch_size'] = [234]
    parameters['max_ciclos'] = [1]
    return parameters


def momentum_test_cases(parameters):
    parameters['taxas_aprendizado'] = [0.1]
    parameters['batch_size'] = [234]
    parameters['max_ciclos'] = [1]
    return parameters


def batch_size_test_cases(parameters):
    parameters['topologias'] = list(product([10, 50, 100, 500, 1000], repeat=1))
    parameters['taxas_aprendizado'] = [0.1]
    parameters['momentum'] = [0.0]
    parameters['max_ciclos'] = [1]
    return parameters


def max_ciclos_test_cases(parameters):
    parameters['topologias'] = list(product([10, 50, 100, 500, 1000], repeat=1))
    parameters['taxas_aprendizado'] = [0.1]
    parameters['momentum'] = [0.0]
    parameters['batch_size'] = [234]
    return parameters


def num_camadas_test_cases(parameters):
    parameters['topologias'] = list(x for y in range(2, 5)
                                    for x in product([10, 50, 100, 500, 1000], repeat=y))

    parameters['taxas_aprendizado'] = [0.1]
    parameters['momentum'] = [0.0]
    parameters['batch_size'] = [234]
    parameters['max_ciclos'] = [1]
    return parameters