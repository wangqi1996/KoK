import sacrebleu


def sort(file, new_hint='D'):
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
                hint = new_hint + hint[1:]
                score = sacrebleu.sentence_bleu(content, [ref[-1].split('\t')[1]]).score
                hypos.append(hint + '\t' + "%.2f" % score + '\t' + content)

    # new_src = [None for _ in range(len(ids))]
    # new_ref = [None for _ in range(len(ids))]
    # new_hypos = [None for _ in range(len(ids))]
    #
    # for i in range(len(ids)):
    #     new_src[ids[i]] = src[i]
    #     new_ref[ids[i]] = ref[i]
    #     new_hypos[ids[i]] = hypos[i]

    return src, ref, hypos


def sort_and_save(file):
    src, ref, hypos = sort(file)
    with open(file + '.sort', 'w') as f:
        for s, r, h in zip(src, ref, hypos):
            f.write(s)
            f.write(r)
            f.write(h)
    print("done")


def combina(baseline_file, knn_file):
    b_src, b_ref, b_hypos = sort(baseline_file, new_hint="base")
    r_src, r_ref, r_hypos = sort(knn_file, new_hint="knn")
    with open(baseline_file + '.combina', 'w') as f:
        for s, r, bh, rh in zip(b_src, b_ref, b_hypos, r_hypos):
            f.write(s)
            f.write(r)
            f.write(bh)
            f.write(rh)
            f.write('\n')
    print("done")


if __name__ == '__main__':
    # base_file = "/home/wangdq/output/output.1457.txt"
    # knn_file = "/home/wangdq/output/output.20059.txt"

    # base_file = "/home/wangdq/output/output.28793.txt"
    # knn_file = "/home/wangdq/output/output.16046.txt"
    base_file = "/home/wangdq/medical/base.50.txt"
    knn_file = "/home/wangdq/medical/kok.a.txt"
    combina(base_file, knn_file)
