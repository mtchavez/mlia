from numpy import *


def load_dataset(filename):
    total = len(open(filename).readline().split('\t')) - 1
    data_matrix = []
    label_matrix = []
    file = open(filename)
    for line in file.readlines():
        line_data = []
        current = line.strip().split('\t')
        for i in range(total):
            line_data.append(float(current[i]))
        data_matrix.append(line_data)
        label_matrix.append(float(current[-1]))
    return data_matrix, label_matrix


def standard_regression(xarr, yarr):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    xtx = xmat.T * xmat
    if linalg.det(xtx) == 0.0:
        print "Matrix is singular, cannot do inverse"
        return
    ws = xtx.I * (xmat .T * ymat)
    return ws


def plot_standard(filename):
    xarr, yarr = load_dataset(filename)
    ws = standard_regression(xarr, yarr)

    xmat = mat(xarr)
    ymat = mat(yarr)
    yhat = xmat * ws

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xmat[:, 1].flatten().A[0], ymat.T[:, 0].flatten().A[0])

    xcopy = xmat.copy()
    xcopy.sort(0)
    yhat = xcopy * ws
    ax.plot(xcopy[:, 1], yhat)
    plt.show()
