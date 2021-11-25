from threading import Thread


def create_thread(test_case, dataset, function):
    return Thread(target=function, args=(test_case, dataset))
