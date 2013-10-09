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
    label_matrix = mat(labels).transpose()
    m, n = shape(data_matrix)
    alpha = 0.001
    cycle_max = 500
    weights = ones((n, 1))
    for k in range(cycle_max):
        h = sigmoid(data_matrix * weights)
        err = (label_matrix - h)
        weights = weights + alpha * data_matrix.transpose() * err
    return weights


def stoc_grad_ascent0(data, labels):
    m, n = shape(data)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(data[i] * weights))
        err = labels[i] - h
        weights = weights + alpha * err * data[i]
    return weights


def plot_best_fit(weights):
    import matplotlib.pyplot as plot
    dm, lm = load_data()
    data_array = array(dm)
    n = shape(data_array)[0]
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(n):
        if int(lm[i]) == 1:
            x1.append(data_array[i, 1])
            y1.append(data_array[i, 2])
        else:
            x2.append(data_array[i, 1])
            y2.append(data_array[i, 2])
    
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='green')
    x = arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plot.xlabel('X1')
    plot.ylabel('X2')
    plot.show()
