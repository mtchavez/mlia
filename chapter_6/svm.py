from numpy import *


def load_data(filename):
    data, labels = [], []
    f = open(filename)
    for l in f.readlines():
        lines = l.strip().split('\t')
        data.append([float(lines[0]), float(lines[1])])
        labels.append(float(lines[2]))
    return data, labels


def select_jrand(i, m):
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


def smo_simple(data, labels, C, toler, max_iter):
    data_matrix = mat(data)
    label_matrix = mat(labels).transpose()
    b = 0
    m, n = shape(data_matrix)
    alphas = mat(zeros((m, 1)))
    iter = 0
    while iter < max_iter:
        alphas_changed = 0
        for i in range(m):
            pred = float(multiply(alphas, label_matrix).T * \
                         (data_matrix * data_matrix[i, :].T)) + b
            err = pred - float(label_matrix[i])
            if ((label_matrix[i]*err < -toler) and (alphas[i] < C)) or \
                    ((label_matrix[i] * err > toler) and (alphas[i] > 0)):
                j = select_jrand(i, m)
                pred2 = float(multiply(alphas, label_matrix).T* \
                              (data_matrix * data_matrix[j, :].T)) + b
                err2 = pred2 - float(label_matrix[j])
                old_ialpha = alphas[i].copy()
                old_jalpha = alphas[j].copy()

                if label_matrix[i] != label_matrix[j]:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])

                if L == H:
                    print "L == H"
                    continue

                eta = 2.0 * data_matrix[i, :] * data_matrix[j, :].T - \
                        data_matrix[i, :] * data_matrix[i, :].T - \
                        data_matrix[j, :] * data_matrix[j, :].T

                if eta >= 0:
                    print "eta >= 0"
                    continue

                alphas[j] -= label_matrix[j] * (err - err2) / eta
                alphas[j] = clip_alpha(alphas[j], H, L)
                if abs(alphas[j] - old_jalpha) < 0.00001:
                    print "j not moving enough"
                    continue

                alphas[i] += label_matrix[j] * label_matrix[i] * \
                        (old_jalpha - alphas[j])
                b1 = b - err - label_matrix[i] * (alphas[i] - old_ialpha) * \
                        data_matrix[i, :] * data_matrix[i, :].T - \
                        label_matrix[j] * (alphas[j] - old_jalpha) * \
                        data_matrix[i, :] * data_matrix[j, :].T
                b2 = b - err2 - label_matrix[i] * (alphas[i] - old_ialpha) * \
                        data_matrix[i, :] * data_matrix[j, :].T - \
                        label_matrix[j] * (alphas[j] - old_jalpha) * \
                        data_matrix[j, :] * data_matrix[j, :].T

                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0

                alphas_changed += 1
                print "iter: %d i: %d, pairs changed %d" % (iter, i, alphas_changed)

        if alphas_changed == 0:
            iter += 1
        else:
            iter = 0
        print "iteration number: %d" % iter
    
    return b, alphas
