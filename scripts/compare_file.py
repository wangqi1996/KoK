import sacrebleu


def sort(file):
    src, hypos, ref, ids = [], [], [], []
    with open(file) as f:
        for line in f:
            if line.startswith('S-'):
                src.append(line)
                ids.append(int(line.split('\t')[0][2:]))
            elif line.startswith('T-'):
                ref.append(line)
            elif line.startswith('D-'):
                hint, score, content = line.split('\t')
                score = sacrebleu.sentence_bleu(content, [ref[-1].split('\t')[1]]).score
                hypos.append(hint + '\t' + "%.2f" % score + '\t' + content)

    new_src = [None for _ in range(len(ids))]
    new_ref = [None for _ in range(len(ids))]
    new_hypos = [None for _ in range(len(ids))]

    for i in range(len(ids)):
        new_src[ids[i]] = src[i]
        new_ref[ids[i]] = ref[i]
        new_hypos[ids[i]] = hypos[i]
    return new_src, new_ref, new_hypos


def sort_and_save(file):
    src, ref, hypos = sort(file)
    with open(file + '.sort', 'w') as f:
        for s, r, h in zip(src, ref, hypos):
            f.write(s)
            f.write(r)
            f.write(h)
    print("done")


def combina(baseline_file, knn_file):
    b_src, b_ref, b_hypos = sort(baseline_file)
    r_src, r_ref, r_hypos = sort(knn_file)
    with open(baseline_file + '.combina', 'w') as f:
        for s, r, bh, rh in zip(b_src, b_ref, b_hypos, r_hypos):
            f.write(s)
            f.write(r)
            f.write(bh)
            f.write(rh)
            f.write('\n')
    print("done")


if __name__ == '__main__':
    base_file = "/home/wangdq/output/b1-nond/generate-train.txt"
    knn_file = "/home/wangdq/output/b1-d/generate-train.txt"
    combina(base_file, knn_file)
