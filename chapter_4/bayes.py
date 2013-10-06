import os
from numpy import *


def load_data():
    posts = [
        ['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classes = [0, 1, 0, 1, 0, 1]
    return posts, classes


def create_vocab(data):
    vocab = set([])
    for doc in data:
        vocab = vocab | set(doc)
    return list(vocab)


def words_to_list(vocab, inputs):
    words = [0] * len(vocab)
    for word in inputs:
        if word in vocab:
            words[vocab.index(word)] = 1
            continue
        print "Word %s not in my vocabulary!" % word
    return words


def bag_of_words_to_list(vocab, inputs):
	words = [0] * len(vocab)
	for word in inputs:
		if word in vocab:
			words[vocab.index(word)] += 1
			continue
	return words


def train(matrix, category):
    total_docs = len(matrix)
    total_words = len(matrix[0])

    abusive_prob = sum(category)/float(total_docs)
    num0_prob = ones(total_words)
    num1_prob = ones(total_words)
    denom0_prob, denom1_prob = 2.0, 2.0

    for i in range(total_docs):
        if category[i] == 1:
            num1_prob += matrix[i]
            denom1_prob += sum(matrix[i])
        else:
            num0_prob += matrix[i]
            denom0_prob += sum(matrix[i])

    p1 = log(num1_prob/denom1_prob)
    p0 = log(num0_prob/denom0_prob)

    return p0, p1, abusive_prob


def classify(inputs, p0, p1, pclass):
    p1 = sum(inputs * p1) + log(pclass)
    p0 = sum(inputs * p0) + log(1.0 - pclass)
    if p1 > p0:
        return 1
    return 0


def test_nb():
    posts, classes = load_data()
    vocab = create_vocab(posts)

    train_matrix = []
    for post in posts:
        words = words_to_list(vocab, post)
        train_matrix.append(words)

    p0, p1, abusive_prob = train(train_matrix, classes)

    test_entry = ['love', 'my', 'dalmation']
    doc = array(words_to_list(vocab, test_entry))
    print test_entry, 'classified as: ', classify(doc, p0, p1, abusive_prob)

    test_entry = ['stupid', 'garbage']
    doc = array(words_to_list(vocab, test_entry))
    print test_entry, 'classified as: ', classify(doc, p0, p1, abusive_prob)


def parse_string(string):
	import re
	tokens = re.split(r'\W*', string)
	return [token.lower() for token in tokens if len(token) > 2]


def test_spam():
	doc_list = []
	class_list = []
	full_text = []
	for i in xrange(1, 26):
		word_list = parse_string(open(os.path.dirname(os.path.realpath(__file__)) + '/email/spam/%d.txt' % i).read())
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(1)
		word_list = parse_string(open(os.path.dirname(os.path.realpath(__file__)) + '/email/ham/%d.txt' % i).read())
		doc_list.append(word_list)
		full_text.extend(word_list)
		class_list.append(0)
	vocab_list = create_vocab(doc_list)
	training_set = range(50)
	test_set = []
	for i in range(10):
		rand_index = int(random.uniform(0,len(training_set)))
		test_set.append(training_set[rand_index])
		del(training_set[rand_index])
	
	train_matrix = []
	train_classes = []
	for doc_index in training_set:
		train_matrix.append(bag_of_words_to_list(vocab_list, doc_list[doc_index]))
		train_classes.append(class_list[doc_index])

	p0, p1, spam_prob = train(array(train_matrix), array(train_classes))
	error_count = 0
	for doc_index in test_set:
		word_vector = bag_of_words_to_list(vocab_list, doc_list[doc_index])
		if classify(array(word_vector), p0, p1, spam_prob) != class_list[doc_index]:
			error_count += 1
	print 'the error rate is: ',float(error_count)/len(test_set)
