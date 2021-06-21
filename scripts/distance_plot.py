import os
import random

import matplotlib.pyplot as plt


def read(file):
    distance = []
    label = []
    with open(file) as f:
        for line in f:
            d, l = line.strip().split('\t')
            distance.append(d)
            label.append(l)

    return distance, label


def distance_plot(file):
    distance, label = read(file)
    x = [random.randint(0, 100) for _ in range(len(distance))]
    colors = ['red', 'green', 'blue', 'purple']

    l0, l1 = [], []
    for d, l in zip(distance, label):
        if l == '0':
            l0.append(d)
        else:
            l1.append(d)
    plt.scatter([random.randint(0, 100) for _ in range(len(l0))], l0, label='0', color='green')
    plt.scatter([random.randint(0, 100) for _ in range(len(l1))], l1, label='1', color='red')
    plt.show()


if __name__ == '__main__':
    dirname = "/Users/wangdq/Documents/code/knn-mt/output"
    for f in os.listdir(dirname):
        distance_plot(os.path.join(dirname, f))
