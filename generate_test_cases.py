from modules.test_cases_generator import generate_test_cases, read_test_case_parts
from modules.test_cases_parameters import *


def main():
    parts = read_test_case_parts()

    # Taxa de aprendizado
    parameters = taxa_aprendizado_test_cases(get_base_parameters())
    generate_test_cases(parameters, parts)

    # Momentum
    parameters = momentum_test_cases(get_base_parameters())
    generate_test_cases(parameters, parts)

    # Batch size
    parameters = batch_size_test_cases(get_base_parameters())
    generate_test_cases(parameters, parts)

    # Numero de ciclos
    parameters = max_ciclos_test_cases(get_base_parameters())
    generate_test_cases(parameters, parts)

    # Numero de camadas e de neurônios
    parameters = num_camadas_test_cases(get_base_parameters())
    generate_test_cases(parameters, parts)


if __name__ == '__main__':
    main()