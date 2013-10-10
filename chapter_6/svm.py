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
