dir="/home/wangdq/data/wmt14-bin/$3/$1"
prefix_file=/home/wangdq/data/split_docs/$2/$1
srclang=de
trglang=en

#BPEROOT=/home/data_ti5_c/wangdq/code/fastBPE/fast
#code=/home/data_ti5_c/wangdq/model/fairseq/pretrain/wmt19/deen/model/ende30k.fastbpe.code
code=/home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq2/code

mkdir -p $dir
for lang in $srclang $trglang; do
  file=$prefix_file.$lang
  subword-nmt apply-bpe -c $code <$file >$dir/bpe.$lang
done

#for lang in $srclang $trglang; do
#  file=$prefix_file.$lang
#  $BPEROOT applybpe $dir/bpe.$lang $file $code
#done

#srcdict=/home/data_ti5_c/wangdq/model/fairseq/pretrain/wmt19/deen/model/dict.$srclang.txt
#trgdict=/home/data_ti5_c/wangdq/model/fairseq/pretrain/wmt19/deen/model/dict.$trglang.txt
srcdict=/home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq2/dict.$srclang.txt
trgdict=/home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq2/dict.$trglang.txt

fairseq-preprocess --source-lang $srclang --target-lang $trglang \
  --trainpref $dir/bpe \
  --destdir $dir \
  --workers 20 \
  --srcdict $srcdict \
  --tgtdict $trgdict
