from numpy import *


def load_dataset(filename):
    features = len(open(filename).readline().split('\t'))
    data_mat = []
    label_mat = []
    file = open(filename)
    for line in file.readlines():
        lines = []
        current = line.strip().split('\t')
        for i in range(features-1):
            lines.append(float(current[i]))
        data_mat.append(lines)
        label_mat.append(float(current[-1]))
    return data_mat, label_mat


def load_simpledata():
    data_matrix = matrix([
        [1., 2.1],
        [2., 1.1],
        [1.3, 1.],
        [1., 1.],
        [2., 1.],
    ])
    labels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_matrix, labels


def stump_classify(data, dimen, threshval, thresineq):
    retarry = ones((shape(data)[0], 1))
    if thresineq == 'lt':
        retarry[data[:, dimen] <= threshval] = -1.0
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


def adaboost_trainDS(data, labels, iters=40):
    weakarr = []
    m = shape(data)[0]
    D = mat(ones((m, 1)) / m)
    agg_class_est = mat(zeros((m, 1)))
    for i in range(iters):
        best_stump, err, class_est = build_stump(data, labels, D)
        print "D: ", D.T
        alpha = float(0.5 * log((1.0 - err) / max(err, 1e-16)))
        best_stump['alpha'] = alpha
        weakarr.append(best_stump)
        print "Class Est: ", class_est.T
        expon = multiply(-1 * alpha * mat(labels).T, class_est)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        agg_class_est += alpha * class_est
        print "Agg Class Est: ", agg_class_est
        agg_errors = multiply(sign(agg_class_est) != mat(labels).T, ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print "Total Error: ", error_rate, "\n"
        if error_rate == 0.0:
            break
    return weakarr


def ada_classify(data, classifier):
    data_matrix = mat(data)
    m = shape(data)[0]
    agg_class_est = mat(zeros((m, 1)))
    for i in range(len(classifier)):
        clss = classifier[i]
        class_est = stump_classify(data_matrix, clss['dim'], clss['thresh'], clss['ineq'])
        agg_class_est += clss['alpha'] * class_est
        print "Agg Class Est: ", agg_class_est
    return sign(agg_class_est)
