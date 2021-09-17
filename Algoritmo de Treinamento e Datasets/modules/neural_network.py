from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning


def create_neural_network(parameters: dict):
    return MLPClassifier(hidden_layer_sizes=parameters['topologia'],
                         activation=parameters['funcao_ativacao'],
                         solver="sgd",
                         learning_rate="constant",
                         learning_rate_init=parameters['taxa_aprendizado'],
                         momentum=parameters['momentum'],
                         max_iter=parameters['max_ciclos'],
                         shuffle=False,
                         batch_size=parameters['batch_size'])


def train_neural_network(neural_network, inputs, outputs):
    status = 'ok'
    try:
        neural_network.fit(inputs, outputs.ravel())

    except ConvergenceWarning:
        status = 'nok'

    return neural_network, status


def test_neural_network(neural_network, inputs, outputs):
    return neural_network.score(inputs, outputs.ravel()) * 100
