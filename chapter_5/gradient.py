from numpy import *


def load_data():
    data = []
    labels = []
    f = open('testSet.txt')
    for line in f.readlines():
        l = line.strip().split()
        data.append([1.0, float(l[0]), float(l[1])])
        labels.append(int(l[2]))
    return data, labels


def sigmoid(x):
    return 1.0 / (1 + exp(-x))


def grad_ascent(data, labels):
    data_matrix = mat(data)
    label_matrix = mat(labels)
    m, n = shape(data_matrix)
    alpha = 0.001
    cycle_max = 500
    weights = ones((n, 1))
    for k in range(cycle_max):
        h = sigmoid(data_matrix * weights)
        err = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * err
    return weights
