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


def lwlr(testpt, xarr, yarr, k=1.0):
    xmat = mat(xarr)
    ymat = mat(yarr).T
    m = shape(xmat)[0]
    weights = mat(eye((m)))
    for j in range(m):
        diffmat = testpt - xmat[j, :]
        weights[j, j] = exp(diffmat * diffmat.T / (-2.0 * k**2))
    xtx = xmat.T * (weights * xmat)
    if linalg.det(xtx) == 0.0:
        print "Matrix is singular, cannot do inverse"
        return
    ws = xtx.I * (xmat.T * (weights * ymat))
    return testpt * ws


def lwlr_test(testarr, xarr, yarr, k=1.0):
    m = shape(testarr)[0]
    yhat = zeros(m)
    for i in range(m):
        yhat[i] = lwlr(testarr[i], xarr, yarr, k)
    return yhat


def plot_lwlr(filename, k=0.001):
    xarr, yarr = load_dataset(filename)
    yhat = lwlr_test(xarr, xarr, yarr, k)
    xmat = mat(xarr)
    srt_ind = xmat[:, 1].argsort(0)
    xsort = xmat[srt_ind][:,0,:]

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xsort[:, 1], yhat[srt_ind])
    ax.scatter(xmat[:, 1].flatten().A[0], mat(yarr).T.flatten().A[0], s=2, c='red')
    plt.show()


def res_err(yarr, yhatarr):
    return ((yarr - yhatarr)**2).sum()
