source scripts/ema/config.sh

export CUDA_VISIBLE_DEVICES=$1

key=$2
filedir=$3
lambda=$4

random_seed=$RANDOM
ref="/home/wangdq/output/ref.$random_seed.txt"
hypos="/home/wangdq/output/hypos.$random_seed.txt"
output="/home/wangdq/output/output.$random_seed.txt"

OUTPUT_PATH=/home/wangdq/output/$random_seed
rm $ref
rm $hypos

bleu_score=""

file_list=$(getFileList $key)
dir=$(getDir $key)
for f in $file_list; do
  bash /home/data_ti5_c/wangdq/code/knn-mt/scripts/ema/knnmt.sh $1 $dir $f $random_seed $lambda
  grep ^T $OUTPUT_PATH/generate-train.txt | cut -f2- >>$ref
  grep ^D $OUTPUT_PATH/generate-train.txt | cut -f3- >>$hypos
  cat $OUTPUT_PATH/generate-train.txt >>$output

  a=$(tail -1 $OUTPUT_PATH/generate-train.txt | grep "BLEU = ....." -o)
  a=${a:7:5}
  bleu_score="$bleu_score\t$a"
done

cat $hypos | sacrebleu $ref
echo $output

python -c "print('$bleu_score')"
mkdir -p /home/wangdq/output/$filedir/
python /home/data_ti5_c/wangdq/code/knn-mt/scripts/recall.py $output
