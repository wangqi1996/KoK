"""
MLP
a: 0.83
b: 0.82
c: 0.82
d: 0.84
e: 0.84
"""
import os.path

import numpy as np
import torch
import torch.nn.functional as f
from torch import nn


def split_dataset(filename, key="a"):
    with open(filename, 'r') as f:
        lines = f.readlines()

    size = len(lines)
    valid_size, test_size = int(size * 0.1), int(size * 0.1)
    train_size = size - valid_size - test_size

    valid_ids = set([i * 10 + 5 for i in range(size // 10)])
    test_ids = set([i * 10 for i in range(size // 10)])
    dim = len(lines[0].strip().split('\t')) - 1

    train_x = np.zeros((train_size, dim), dtype=float)
    train_y = np.zeros((train_size, 1), dtype=int)
    valid_x = np.zeros((valid_size, dim), dtype=float)
    valid_y = np.zeros((valid_size, 1), dtype=int)
    test_x = np.zeros((test_size, dim), dtype=float)
    test_y = np.zeros((test_size, 1), dtype=int)

    train_index, valid_index, test_index = 0, 0, 0
    for id, sample in enumerate(lines):
        items = [float(i) for i in sample.strip().split('\t')]
        assert not (id in valid_ids and id in test_ids)
        if id in valid_ids:
            valid_x[valid_index] = items[:-1]
            valid_y[valid_index] = int(items[-1])
            valid_index += 1
        elif id in test_ids:
            test_x[test_index] = items[:-1]
            test_y[test_index] = int(items[-1])
            test_index += 1
        else:
            train_x[train_index] = items[:-1]
            train_y[train_index] = int(items[-1])
            train_index += 1

    assert train_index == train_x.shape[0]
    assert valid_size == valid_x.shape[0]
    assert test_size == test_x.shape[0]
    # save dataset
    filename = os.path.dirname(filename) + '/' + key
    os.makedirs(filename, exist_ok=True)
    np.save(os.path.join(filename, "train_x"), train_x)
    np.save(os.path.join(filename, "train_y"), train_y)
    np.save(os.path.join(filename, "valid_x"), valid_x)
    np.save(os.path.join(filename, "valid_y"), valid_y)
    np.save(os.path.join(filename, "test_x"), test_x)
    np.save(os.path.join(filename, "test_y"), test_y)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return len(self.data_x)


def get_loss(output, labels):
    loss = f.binary_cross_entropy_with_logits(output, labels, reduction="none").mean()
    return loss


def load_dataset(dirname, batch_size):
    train_x_name = os.path.join(dirname, "train_x.npy")
    train_y_name = os.path.join(dirname, "train_y.npy")
    valid_x_name = os.path.join(dirname, "valid_x.npy")
    valid_y_name = os.path.join(dirname, "valid_y.npy")
    test_x_name = os.path.join(dirname, "test_x.npy")
    test_y_name = os.path.join(dirname, "test_y.npy")
    train_x = torch.Tensor(np.load(train_x_name)).cuda()
    train_y = torch.Tensor(np.load(train_y_name)).cuda()
    valid_x = torch.Tensor(np.load(valid_x_name)).cuda()
    valid_y = torch.Tensor(np.load(valid_y_name)).cuda()
    test_x = torch.Tensor(np.load(test_x_name)).cuda()
    test_y = torch.Tensor(np.load(test_y_name)).cuda()

    print("training data size: ", train_x.size())

    dim = train_x.size(-1)

    # count label 0 and label 1
    def count_label(dataset, key):
        label_0 = (dataset == 0).long().sum().item()
        label_1 = (dataset == 1).long().sum().item()
        print(key, ": label_0: ", label_0 / (label_0 + label_1), " label_1: ", label_1 / (label_0 + label_1))

    count_label(train_y, "train")
    count_label(valid_y, "valid")
    count_label(test_y, "test")
    train_dataloader = torch.utils.data.DataLoader(Dataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(Dataset(valid_x, valid_y), batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(Dataset(test_x, test_y), batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader, test_dataloader, dim


def load_model(input_dim, hidden_dim=0):
    # 做一个二分类
    # model = nn.Sequential(
    #     nn.Linear(input_dim, hidden_dim),
    #     nn.GELU(),
    #     nn.Dropout(0.3),
    #     nn.Linear(hidden_dim, 1),
    # ).cuda()
    #
    # """ init the model parameters"""
    # nn.init.xavier_normal_(model[0].weight)
    # nn.init.xavier_normal_(model[-1].weight)

    model = nn.Linear(input_dim, 1).cuda()
    nn.init.uniform(model.weight)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    return model, optimizer


def compute_accuracy(predict, reference):
    correct, all = 0, 0
    for p, r in zip(predict, reference):
        all += 1
        if p == r:
            correct += 1
    return float("%.2f" % (correct / all))


def get_predict(output):
    predict = (output.sigmoid() > 0.5).long()
    return predict


def eval(model, dataloader):
    predict_list = []
    reference_list = []
    model.eval()
    with torch.no_grad():
        for j, data in enumerate(dataloader):
            inputs, labels = data
            output = model(inputs)
            loss = get_loss(output, labels)
            predict = get_predict(output)
            predict_list.extend(predict.cpu().tolist())
            reference_list.extend(labels.cpu().tolist())

    accuracy = compute_accuracy(predict_list, reference_list)
    return accuracy, loss.item()


if __name__ == '__main__':
    key = 'e'
    split_dataset("/home/wangdq/lambda-datastore-W/key." + key + ".txt", key=key)

    """
    无用的参数：
    batch-size，
    # """
    # epoch, batch_size = 100, 32
    # max_accuracy, patience, max_patience = 0, 0, 300
    #
    # dirname = "/home/wangdq/lambda-datastore/"
    #
    # train_dataloader, valid_dataloader, test_dataloader, dim = load_dataset(dirname, batch_size=batch_size)
    # model, optimizer = load_model(dim)
    # #
    # for e in range(epoch):
    #
    #     for i, data in enumerate(train_dataloader):
    #         model.train()
    #         optimizer.zero_grad()
    #         inputs, labels = data
    #
    #         output = model(inputs)
    #         loss = get_loss(output, labels)
    #         loss.backward()
    #         optimizer.step()
    #
    #         if i % 30 == 0:
    #             accuracy, valid_loss = eval(model, valid_dataloader)
    #             print(str(e), "-", str(i), " train-loss: ", loss.item(), " valid-loss: ", valid_loss, ", accuracy: ",
    #                   accuracy)
    #             if accuracy < max_accuracy:
    #                 patience += 1
    #                 if patience > max_patience:
    #                     exit(0)
    #             if accuracy > max_accuracy:
    #                 max_accuracy = accuracy
    #                 accuracy, loss = eval(model, test_dataloader)
    #                 print("test-set: ", " loss: ", loss, ", accuracy: ", accuracy)
    #                 torch.save(model.state_dict(), os.path.join(dirname, "model.pt"))
