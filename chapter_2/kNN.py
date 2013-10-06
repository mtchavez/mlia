from numpy import *
from os import listdir
import operator


def create_dataset():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(input, data, labels, k):
    data_size = data.shape[0]
    diff_mat = tile(input, (data_size, 1)) - data
    sq_diff_mat = diff_mat**2
    distances = sq_diff_mat.sum(axis=1)**0.5
    sorted_dist = distances.argsort()
    class_count = {}
    for i in range(k):
        label_i = labels[sorted_dist[i]]
        class_count[label_i] = class_count.get(label_i, 0) + 1
    sorted_class_count = sorted(
        class_count.iteritems(),
        key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def file_to_matrix(filename):
    f = open(filename)
    lines = len(f.readlines())
    matrix = zeros((lines, 3))
    labels = []
    f = open(filename)
    i = 0
    for line in f.readlines():
        line = line.strip()
        example = line.split('\t')
        matrix[i:] = example[0:3]
        labels.append(example[-1])
        i += 1
    return matrix, labels


def normalize(data):
    min_vals = data.min(0)
    max_vals = data.max(0)
    ranges = max_vals - min_vals
    norm_data = zeros(shape(data))
    m = data.shape[0]
    norm_data = data - tile(min_vals, (m, 1))
    norm_data = norm_data/tile(ranges, (m, 1))
    return norm_data, ranges, min_vals


def dating_class_test():
    ratio = 0.05
    data, labels = file_to_matrix('dating_set.txt')
    norm, ranges, min_vals = normalize(data)
    m = norm.shape[0]
    test_examples = int(m*ratio)
    error = 0.0
    for i in range(test_examples):
        result = classify0(norm[i, :], norm[test_examples:m, :], labels[test_examples:m], 3)
        print "the classifier came back with: %r, the real answer is: %r" % (result, labels[i])
        if (result != labels[i]):
            error += 1.0
    print "the total error rate is: %f" % (error/float(test_examples))


def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))
