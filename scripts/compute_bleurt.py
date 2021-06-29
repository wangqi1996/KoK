import os

import sacrebleu
import bleurt
from bleurt import score

checkpoint = "/home/wangdq/bleurt/model/bleurt-base-128/"
scorer = score.BleurtScorer(checkpoint)

dirname = "/home/wangdq/output/useful/"
filename1 = ['distance.50.txt', 'distance.100.txt', 'distance.200.txt', 'distance.500.txt', 'distance.1000.txt']
filename2 =  ['lambda.50.txt', 'lambda.100.txt', 'lambda.200.txt', 'lambda.500.txt', 'lambda.1000.txt']
filename3 = ['knn3.50.txt', 'knn3.100.txt', 'knn3.200.txt', 'knn3.500.txt', 'knn3.1000.txt']
filename4 = ['tune.50.txt', 'tune.100.txt', 'tune.200.txt', 'tune.500.txt', 'tune.1000.txt']
filename5 = ['base.50.txt', 'base.100.txt', 'base.200.txt', 'base.500.txt', 'base.1000.txt']

for filename in [filename1,filename2,filename3,filename4,filename5]:
    bleurt_score = ''
    bleu = ''
    for f in filename:

        file = os.path.join(dirname, f)
        print("-----%s-------" % file)

        os.system("grep ^T %s | cut -f2- > /home/wangdq/ref" % file)
        os.system("grep ^D %s | cut -f3- > /home/wangdq/hypo" % file)
        bleu_score = "%.1f" % sacrebleu.corpus_bleu(open("/home/wangdq/hypo"), [open("/home/wangdq/ref")]).score
        bleu += (bleu_score + '\t')

        references = [l.strip() for l in open("/home/wangdq/ref")]
        candidates = [l.strip() for l in open("/home/wangdq/hypo")]

        scores = scorer.score(references=references, candidates=candidates)
        scores = sum(scores) / len(scores)
        bleurt_score += ("%.2f" % (scores* 100) + '\t')
    print(bleu)
    print(bleurt_score)
