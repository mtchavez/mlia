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


# Full Plat SMO

class optStruct:

    def __init__(self, data, labels, C, toler):
        self.X = data
        self.labels = labels
        self.C = C
        self.tol = toler
        self.m = shape(data)[0]
        self.alphas = mat(zeros((self.m, 1)))
        self.b = 0
        self.err_cache = mat(zeros((self.m, 1)))


def calc_kerr(oS, k):
    pred = float(multiply(oS.alphas, oS.labels).T * \
            (oS.X * oS.X[k, :].T)) + oS.b
    return pred - float(oS.labels[k])


def select_j(i, oS, ierr):
    max_k = -1
    max_delta = 0
    jerr = 0
    oS.err_cache[i] = ierr
    valid_ecache = nonzero(oS.err_cache[:, 0].A)[0]
    if len(valid_ecache) > 1:
        for k in valid_ecache:
            if k == i:
                continue
            kerr = calc_kerr(oS, k)
            edelta = abs(ierr - kerr)
            if edelta > max_delta:
                max_k = k
                max_delta = edelta
                jerr = kerr
        return max_k, jerr
    else:
        j = select_jrand(i, oS.m)
        jerr = calc_kerr(oS, j)
    return j, jerr


def update_kerr(oS, k):
    kerr = calc_kerr(oS, k)
    oS.err_cache[k] = kerr


def inner_l(i, oS):
    ierr = calc_kerr(oS, i)
    if ((oS.labels[i] * ierr < -oS.tol) and (oS.alphas[i] < oS.C)) or \
            ((oS.labels[i] * ierr > oS.tol) and (oS.alphas[i] > 0)):
        j, jerr = select_j(i, oS, ierr)
        old_ialpha = oS.alphas[i].copy()
        old_jalpha = oS.alphas[j].copy()
        if oS.labels[i] != oS.labels[j]:
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])

        if L == H:
            print "L == H"
            return 0

        eta = 2.0 * oS.X[i, :] * oS.X[j, :].T - oS.X[i, :] * oS.X[i, :].T - \
                oS.X[j, :] * oS.X[j, :].T

        if eta >= 0:
            print "eta >= 0"
            return 0

        oS.alphas[j] -= oS.labels[j] * (ierr - jerr) / eta
        oS.alphas[j] = clip_alpha(oS.alphas[j], H, L)
        update_kerr(oS, j)

        if abs(oS.alphas[j] - old_jalpha) < 0.00001:
            print "j not moving enough"
            return 0

        oS.alphas[i] += oS.labels[j] * oS.labels[i] * (old_jalpha - oS.alphas[j])
        update_kerr(oS, i)

        b1 = oS.b - ierr - oS.labels[i] * (oS.alphas[i] - old_ialpha) * \
                oS.X[i, :] * oS.X[i,:].T - oS.labels[j] * \
                (oS.alphas[j] - old_jalpha) * oS.X[i, :] * oS.X[j, :].T
        
        b2 = oS.b - jerr - oS.labels[i] * (oS.alphas[i] - old_ialpha) * \
                oS.X[i, :] * oS.X[j, :].T - oS.labels[j] * \
                (oS.alphas[j] - old_jalpha) * oS.X[j, :] * oS.X[j, ].T

        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        
        return 1
    else:
        return 0


def smo_full(data, labels, C, toler, max_iter, ktup=('lin', 0)):
    oS = optStruct(mat(data), mat(labels).transpose(), C, toler)
    iter = 0
    entire_set = True
    alphas_changed = 0
    while iter < max_iter and (alphas_changed > 0 or entire_set):
        alphas_changed = 0
        if entire_set:
            for i in range(oS.m):
                alphas_changed += inner_l(i, oS)
            print "Full Set, iter: %d i: %d, pairs changed %d" % (iter, i, alphas_changed)
            iter += 1
        else:
            nonbound = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonbound:
                alphas_changed += inner_l(i, oS)
                print "non bound, iter: %d i: %d, pairs changed %d" % (iter, i, alphas_changed)
            iter += 1

        if entire_set:
            entire_set = False
        elif alphas_changed == 0:
            entire_set = True

        print "iteration: %d" % iter

    return oS.b, oS.alphas


def calc_ws(alphas, data, labels):
    X = mat(data)
    label_matrix = mat(labels).transpose()
    m, n = shape(X)
    w = zeros((n, 1))
    for i in range(m):
        w += multiply(alphas[i] * label_matrix[i], X[i, :].T)
    return w
