from math import log
import operator


def createDataSet():
    dataSet = [
        [1, 1, 'yes'],
        [1, 1, 'yes'],
        [1, 0, 'no'],
        [0, 1, 'no'],
        [0, 1, 'no']
    ]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels


def entropy(data):
    entries = len(data)
    counts = {}
    for entry in data:
        label = entry[-1]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    ent = 0.0
    for key in counts:
        prob = float(counts[key])/entries
        ent -= prob * log(prob, 2)
    return ent


def split(data, label, value):
    new_data = []
    for entry in data:
        if entry[label] == value:
            reduced_entry = entry[:label]
            reduced_entry.extend(entry[label+1:])
            new_data.append(reduced_entry)
    return new_data


def best_feature(data):
    total = len(data[0]) - 1
    base_entropy = entropy(data)
    max_info = 0.0
    best_feature = -1
    for i in range(total):
        features = [example[i] for example in data]
        new_entropy = 0.0
        for value in set(features):
            sub_data = split(data, i, value)
            prob = len(sub_data)/float(len(data))
            new_entropy += prob * entropy(sub_data)
        info_gain = base_entropy - new_entropy
        if (info_gain > max_info):
            max_info = info_gain
            best_feature = i
    return best_feature


def majority(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
        class_count[vote] += 1
    sorted_class_count = sorted(class_count.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def create_tree(data, labels):
    class_list = [example[-1] for example in data]
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    if len(data[0]) == 1:
        return majority(class_list)

    best_feat = best_feature(data)
    best_label = labels[best_feat]

    tree = {best_label: {}}
    del(labels[best_feat])

    values = [ex[best_feat] for ex in data]
    unique_vals = set(values)

    for value in unique_vals:
        sub_labels = labels[:]
        tree[best_label][value] = create_tree(split(data, best_feat, value), sub_labels)

    return tree


def classify(tree, labels, test_feature):
    first_label = tree.keys()[0]
    first_decision = tree[first_label]
    feat_index = labels.index(first_label)
    for key in first_decision.keys():
        if test_feature[feat_index] == key:
            if type(first_decision[key]).__name__ == 'dict':
                class_label = classify(first_decision[key], labels, test_feature)
            else:
                class_label = first_decision[key]
    return class_label


def store_tree(tree, filename):
    import pickle
    fw = open(filename, 'w')
    pickle.dump(tree, fw)
    fw.close()


def restore_tree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)
