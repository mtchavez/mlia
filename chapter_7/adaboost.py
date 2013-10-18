from numpy import *


def load_simpledata():
    data_matrix = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.],
    ])
    labels = [1.0, 1.0, -1.0, 1.0]
    return data_matrix, labels


def stump_classify(data, dimen, threshval, thresineq):
    retarry = ones((shape(data)[0], 1))
    if thresineq == 'lt':
        retarry[data[:, dimen] < threshval] = 1.0
    else:
        retarry[data[:, dimen] > threshval] = -1.0
    return retarry


def build_stump(data, labels, D):
    data = mat(data)
    labels = mat(labels).T
    m, n = shape(data)
    steps = 10.0
    best_stump = {}
    best_class = mat(zeros((m, 1)))
    min_err = inf
    for i in range(n):
        rangemin = data[:, i].min()
        rangemax = data[:, i].max()
        stepsize = (rangemax - rangemin) / steps

        for j in range(-1, int(steps) + 1):
            for inequal in ['lt', 'gt']:
                threshval = (rangemin + float(j) * stepsize)
                predicted = stump_classify(data, i, threshval, inequal)
                errarr = mat(ones((m, 1)))
                errarr[predicted == labels] = 0
                weighted_err = D.T * errarr

                if weighted_err < min_err:
                    min_err = weighted_err
                    best_class = predicted.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = threshval
                    best_stump['ineq'] = inequal

    return best_stump, min_err, best_class
