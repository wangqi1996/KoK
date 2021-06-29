"""
svm
"""

import os

import numpy as np
from sklearn import svm

from scripts.simple_mlp import split_dataset


def load_dataset(dirname):
    train_x_name = os.path.join(dirname, "train_x.npy")
    train_y_name = os.path.join(dirname, "train_y.npy")
    valid_x_name = os.path.join(dirname, "valid_x.npy")
    valid_y_name = os.path.join(dirname, "valid_y.npy")
    test_x_name = os.path.join(dirname, "test_x.npy")
    test_y_name = os.path.join(dirname, "test_y.npy")
    train_x = np.load(train_x_name)
    train_y = np.load(train_y_name).squeeze()
    valid_x = np.load(valid_x_name)
    valid_y = np.load(valid_y_name).squeeze()
    test_x = np.load(test_x_name)
    test_y = np.load(test_y_name).squeeze()

    print("training data size: ", train_x.shape)

    dim = train_x.shape[-1]

    # count label 0 and label 1
    def count_label(dataset, key):
        label_0 = (dataset == 0).sum()
        label_1 = (dataset == 1).sum()
        print(key, ": label_0: ", label_0 / (label_0 + label_1), " label_1: ", label_1 / (label_0 + label_1))

    count_label(train_y, "train")
    count_label(valid_y, "valid")
    count_label(test_y, "test")
    return dim, train_x, train_y, valid_x, valid_y, test_x, test_y


def compute_accuracy(predict, reference):
    correct, all = 0, 0
    for p, r in zip(predict, reference):
        all += 1
        if p == r:
            correct += 1
    return float("%.2f" % (correct / all))


def get_predict(output):
    predict = (output.sigmoid() > 0.5)
    return predict


def predict(model, x, y):
    output = model.predict(x)
    correct = (output == y).sum()
    all = output.size
    accuracy = "%.2f" % (correct / all)
    print("all accuracy: ", accuracy)

    def count_label(label):
        correct = ((output == label) & (y == label)).sum()
        recall = (y == label).sum()
        accuracy = (output == label).sum()
        print(label, " accuracy: ", "%.2f" % (correct / accuracy))
        print(label, " recall: ", "%.2f" % (correct / recall))

    count_label(0)
    count_label(1)


def train():
    key = "e"
    split_dataset("/home/wangdq/lambda-datastore-nmt/key.%s.txt" % key, key=key)
    dirname = "/home/wangdq/lambda-datastore-nmt/" + key
    dim, train_x, train_y, valid_x, valid_y, test_x, test_y = load_dataset(dirname)
    clf = svm.SVC()
    clf.fit(train_x, train_y)

    predict(clf, valid_x, valid_y)
    predict(clf, test_x, test_y)

    import pickle
    filename = os.path.join(dirname, "svm.pt")
    pickle.dump(clf, open(filename, "wb"))


if __name__ == '__main__':
    train()
