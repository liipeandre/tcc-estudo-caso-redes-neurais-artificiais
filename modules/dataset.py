from sys import argv

from numpy import loadtxt, uint8, float16
from sklearn.preprocessing import StandardScaler


def load_test_cases():
    with open(argv[1], "r") as file:
        return file.readlines()


def load_dataset(filename):
    return loadtxt(filename, dtype=uint8, delimiter=";")


def split_dataset(dataset):
    # Primeira coluna do dataset é a saída e as demais são as entradas
    inputs = dataset[:, 1:]
    outputs = dataset[:, 0]

    return inputs, outputs


def normalize_dataset(inputs):
    normalizer = StandardScaler()
    normalizer.fit(inputs)

    return normalizer.transform(inputs).astype(float16)
