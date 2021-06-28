import sacrebleu
import stopwordsiso as stopwords
from sacremoses import MosesTokenizer

"""
注意事项：
1. 对文本进行了tokenizer处理 
2. 判断是否是stopwords，a) 对输入全部进行小写处理  b)纯数字  c) 标点符号
3. 单个文件计算，将recall和all合起来计算召回率
4.
5.
"""
import string

punc = set(string.punctuation)

import re

pattern = re.compile(r"^\d+\.?\d*$")


def in_stopwords(token: str, stoplist):
    if token.lower() in stoplist:
        return True
    if token in punc:
        return True
    if pattern.match(token) is not None:
        return True

    return False


import os


def compute_bleu(file):
    os.system("grep ^T %s | cut -f2- > ref" % file)
    os.system("grep ^D %s | cut -f3- > hypo" % file)
    return "%.1f" % sacrebleu.corpus_bleu(open("hypo"), [open("ref")]).score


def time_index(times):
    if times < 2:
        return times
    elif 2 <= times <= 5:
        return 2
    elif 6 <= times <= 9:
        return 3
    else:
        return 4


def recall(file):
    tok = MosesTokenizer(lang="en")
    max_r = 5  # time_index的个数
    base_recall = [0 for _ in range(max_r)]
    knn_recall = [0 for _ in range(max_r)]
    all = [0 for _ in range(max_r)]
    stoplist = stopwords.stopwords("en")
    occurrence = {}  # count set  ref
    content = []
    with open(file) as f:
        ref, hypos = None, None
        for line in f:
            if line.startswith("S-"):
                content.append(line)
            elif line.startswith("T-"):
                ref = list(set(tok.tokenize(' '.join(line.split('\t')[1:]))))
                id = int(line.split('\t')[0][2:])
                if id == 0:
                    occurrence = {}
                content.append(line)
            elif line.startswith("base-"):
                base_hypos = set(tok.tokenize(' '.join(line.split('\t')[2:])))
                base_line = line
            elif line.startswith("knn-"):
                knn_hypos = set(tok.tokenize(' '.join(line.split('\t')[2:])))
                r0_str = "R0:"
                base_r0, knn_r0, all_r0 = 0, 0, 0
                for index, r_token in enumerate(ref):
                    if in_stopwords(r_token, stoplist):
                        continue
                    times = occurrence.get(r_token, 0)
                    occurrence.setdefault(r_token, 0)
                    occurrence[r_token] += 1

                    if times == 0:
                        r0_str += (" " + r_token)
                        all_r0 += 1

                    t_index = time_index(times)
                    all[t_index] += 1
                    if r_token in base_hypos:
                        base_recall[t_index] += 1
                        base_r0 = base_r0 + 1 if times == 0 else base_r0

                    if r_token in knn_hypos:
                        knn_recall[t_index] += 1
                        knn_r0 = knn_r0 + 1 if times == 0 else knn_r0
                base_r0 = base_r0 / all_r0 if all_r0 != 0 else 0
                knn_r0 = knn_r0 / all_r0 if all_r0 != 0 else 0
                if base_r0 > knn_r0:
                    content.append(r0_str + '\n')

                    base = base_line.split('\t')
                    base_str = base[0] + "\t" + base[1] + '\t' + "%.2f" % base_r0 + '\t' + base[-1]
                    content.append(base_str)

                    knn = line.split('\t')
                    knn_str = knn[0] + "\t" + knn[1] + '\t' + "%.2f" % knn_r0 + '\t' + knn[-1]
                    content.append(knn_str + '\n')
                else:
                    content.pop()
                    content.pop()

                ref = None

    for r, a in zip(base_recall, all):
        print("%.2f" % (r / a * 100), end="\t")
    print()
    for r, a in zip(knn_recall, all):
        print("%.2f" % (r / a * 100), end="\t")
    print()
    for a in all:
        print(a, end='\t')
    print()
    with open(file + ".R0", 'w') as f:
        f.writelines(content)


if __name__ == '__main__':
    # import sys
    #
    # file = sys.argv[1]
    # file = "/home/wangdq/output/output.1457.txt.combina"
    file = "/home/wangdq/output/output.28793.txt.combina"
    recall(file)

"""
knn:
a: /home/wangdq/output/output.1306.txt
b: /home/wangdq/output/output.20059.txt
c: /home/wangdq/output/output.4157.txt
d: /home/wangdq/output/output.31.txt
e: /home/wangdq/output/output.3257.txt

baseline:
a: 
b:
c:
d:
e: 
"""
