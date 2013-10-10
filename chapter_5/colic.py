import gradient as grad
from numpy import *


def classify(x, weights):
    prob = grad.sigmoid(sum(x*weights))
    return 1.0 if prob > 0.5 else 0.0


def colic_test():
    horse_train = open('horseColicTraining.txt')
    horse_test = open('horseColicTest.txt')
    train_set = []
    train_labels = []
    for line in horse_train.readlines():
        l = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(l[i]))
        train_set.append(line_array)
        train_labels.append(float(l[21]))

    train_weights = grad.stoc_grad_ascent1(array(train_set), train_labels, 500)
    error_count = 0
    test_vector_count = 0.0
    for line in horse_test.readlines():
        test_vector_count += 1.0
        l = line.strip().split('\t')
        line_array = []
        for i in range(21):
            line_array.append(float(l[i]))
        if int(classify(array(line_array), train_weights)) != int(l[21]):
            error_count += 1

    error_rate = (float(error_count)/ test_vector_count)
    print "The error rate is: %f" % error_rate
    return error_rate


def multitest():
    total_tests = 10
    error_sum = 0.0
    for k in range(total_tests):
        error_sum += colic_test()
    print "After %d iterations the avg error rate is %f" % (total_tests, error_sum / float(total_tests))
