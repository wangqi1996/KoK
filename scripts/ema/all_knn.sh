export CUDA_VISIBLE_DEVICES=$1

key=$2
lambda=$3

random_seed=$RANDOM
ref="/home/wangdq/output/ref.$random_seed.txt"
hypos="/home/wangdq/output/hypos.$random_seed.txt"
output="/home/wangdq/output/output.$random_seed.txt"

OUTPUT_PATH=/home/wangdq/output/$random_seed
rm $ref
rm $hypos

bleu_score=""

if [ $key == "a" ]; then
  a1=H-679-IV-de
  a2=H-717-Annex-de
  a3=lamictal_bi_de
  a4=menitorix_bi_de
  a5=Doxyprex_Background_Information-de
  a6=norfloxacin-bi-de
  a7=uman_big_q_a_de
  a8=sabumalin_q_a_de
  a9=etoricoxib-arcoxia-bi-de
  a10=sanohex_q_a_de
  a11=implanon_Q_A_de
  a12=emea-2006-0258-00-00-ende
  a13=V-121-de1
  a14=H-902-WQ_A-de
  a15=V-141-de1
  a16=V-137-de1
  a17=Hexavac-H-298-Z-28-de
  a18=H-891-de1
  a19=V-030-de1
  a20=V-107-de1
  a21=compagel-v-a-33-030-de
  a22=V-126-de1

  for f in $a1 $a2 $a3 $a4 $a5 $a6 $a7 $a8 $a9 $a10 $a11 $a12 $a13 $a14 $a15 $a16 $a17 $a18 $a19 $a20 $a21 $a22; do
    bash /home/data_ti5_c/wangdq/code/knn-mt/scripts/ema/knnmt.sh $1 50 $f $random_seed $lambda
    grep ^T $OUTPUT_PATH/generate-train.txt | cut -f2- >>$ref
    grep ^D $OUTPUT_PATH/generate-train.txt | cut -f3- >>$hypos
    cat $OUTPUT_PATH/generate-train.txt >>$output
    a=$(tail -1 $OUTPUT_PATH/generate-train.txt | grep "BLEU = ....." -o)
    a=${a:7:5}
    bleu_score="$bleu_score\t$a"
  done
elif [ $key == "b" ]; then
  b1=093604de1
  b2=H-741-de1
  b3=H-897-de1
  b4=H-933-de1
  b5=tritazide_q_a_de
  b6=V-133-de1
  b7=H-915-de1
  b8=H-725-de1
  b9=400803de1
  b10=H-890-de1
  b11=Veralipride-H-A-31-788-de
  b12=Belanette-AnnexI-III-de
  b13=V-A-35-029-de
  b14=112901de4

  for f in $b1 $b2 $b3 $b4 $b5 $b6 $b7 $b8 $b9 $b10 $b11 $b12 $b13 $b14; do
    bash /home/data_ti5_c/wangdq/code/knn-mt/scripts/ema/knnmt.sh $1 100 $f $random_seed $lambda
    grep ^T $OUTPUT_PATH/generate-train.txt | cut -f2- >>$ref
    grep ^D $OUTPUT_PATH/generate-train.txt | cut -f3- >>$hypos
    cat $OUTPUT_PATH/generate-train.txt >>$output
    a=$(tail -1 $OUTPUT_PATH/generate-train.txt | grep "BLEU = ....." -o)
    a=${a:7:5}
    bleu_score="$bleu_score\t$a"
  done
elif [ $key == "c" ]; then

  c1=49533907de
  c2=implanon_annexI_IV_de
  c3=EMEA-CVMP-82633-2007-de
  c4=sanohex_annexI_III_de
  c5=V-048-PI-de
  c6=V-041-PI-de
  c7=V-047-PI-de

  for f in $c1 $c2 $c3 $c4 $c5 $c6 $c7; do
    bash /home/data_ti5_c/wangdq/code/knn-mt/scripts/ema/knnmt.sh $1 200 $f $random_seed $lambda
    grep ^T $OUTPUT_PATH/generate-train.txt | cut -f2- >>$ref
    grep ^D $OUTPUT_PATH/generate-train.txt | cut -f3- >>$hypos
    cat $OUTPUT_PATH/generate-train.txt >>$output
    a=$(tail -1 $OUTPUT_PATH/generate-train.txt | grep "BLEU = ....." -o)
    a=${a:7:5}
    bleu_score="$bleu_score\t$a"
  done
elif [ $key == "d" ]; then

  d1=V-105-PI-de
  d2=H-391-PI-de
  d3=H-668-PI-de
  d4=H-960-PI-de

  for f in $d1 $d2 $d3 $d4; do
    bash /home/data_ti5_c/wangdq/code/knn-mt/scripts/ema/knnmt.sh $1 500 $f $random_seed $lambda
    grep ^T $OUTPUT_PATH/generate-train.txt | cut -f2- >>$ref
    grep ^D $OUTPUT_PATH/generate-train.txt | cut -f3- >>$hypos
    cat $OUTPUT_PATH/generate-train.txt >>$output
    a=$(tail -1 $OUTPUT_PATH/generate-train.txt | grep "BLEU = ....." -o)
    a=${a:7:5}
    bleu_score="$bleu_score\t$a"
  done
elif [ $key == "e" ]; then
  e1=H-884-PI-de
  e2=H-287-PI-de
  e3=H-890-PI-de
  e4=H-273-PI-de
  e5=H-115-PI-de
  for f in $e1 $e2 $e3 $e4 $e5; do
    bash /home/data_ti5_c/wangdq/code/knn-mt/scripts/ema/knnmt.sh $1 1000 $f $random_seed $lambda
    grep ^T $OUTPUT_PATH/generate-train.txt | cut -f2- >>$ref
    grep ^D $OUTPUT_PATH/generate-train.txt | cut -f3- >>$hypos
    cat $OUTPUT_PATH/generate-train.txt >>$output
    a=$(tail -1 $OUTPUT_PATH/generate-train.txt | grep "BLEU = ....." -o)
    a=${a:7:5}
    bleu_score="$bleu_score\t$a"
  done
elif [ $key == "f" ]; then
  f1=H-116-PI-de
  f2=H-481-PI-de
  f3=H-250-PI-de
  f4=H-088-PI-de
  f5=H-333-PI-de
  f6=H-282-PI-de
  for f in $f1 $f2 $f3 $f4 $f5 $f6; do
    bash /home/data_ti5_c/wangdq/code/knn-mt/scripts/ema/knnmt.sh $1 1000 $f $random_seed $lambda
    grep ^T $OUTPUT_PATH/generate-train.txt | cut -f2- >>$ref
    grep ^D $OUTPUT_PATH/generate-train.txt | cut -f3- >>$hypos
    cat $OUTPUT_PATH/generate-train.txt >>$output
    a=$(tail -1 $OUTPUT_PATH/generate-train.txt | grep "BLEU = ....." -o)
    a=${a:7:5}
    bleu_score="$bleu_score\t$a"
  done
fi
cat $hypos | sacrebleu $ref
echo $output
echo $bleu_score

python -c "print('$bleu_score')"
