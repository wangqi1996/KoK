def split():
    split = "train"
    filename = "/home/wangdq/wmt14/retrieve/" + split + ".mem"
    content = [[] for _ in range(5)]
    with open(filename) as f:
        for line in f:
            src, tgt, *mem = line.strip().split('\t')
            for i in range(5):
                content[i].append(mem[i * 2] + '\n')
    for i in range(5):
        with open(filename + '.' + str(i), 'w') as f:
            f.writelines(content[i])


if __name__ == '__main__':
    split()

# id=4
# fairseq-preprocess --source-lang en  --target-lang de \
#   --trainpref train.$id \
#   --testpref test.$id \
# --validpref valid.$id \
#   --destdir data-bin-$id/ \
#   --workers 20 \
#   --srcdict /home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq2/dict.en.txt --only-source
