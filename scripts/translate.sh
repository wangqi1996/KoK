export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GUN
rm -rf $LOGDIR

python fairseq_cli/generate.py /home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq2 \
  --gen-subset $3 \
  --seed 1234 \
  --task translation \
  --batch-size 128 \
  --beam 4 \
  --remove-bpe \
  -s de -t en \
  --path /home/wangdq/save/wmt14/checkpoint_last.pt \
  --arch transformer_wmt_en_de_big \
  --max-len-a 1.2 \
  --max-len-b 10 \
  --results-path $2 \
  --model-overrides "{'valid_subset': '$3'}"

tail -1 $2/generate-$3.txt
#bash ../new/scripts/compound_split_bleu.sh $2/generate-$3.txt

# best: Generate test with beam=4: BLEU4 = 32.52, 65.9/40.1/26.6/18.2 (BP=0.968, ratio=0.968, syslen=66762, reflen=68951)
# ave_best:Generate test with beam=4: BLEU4 = 32.56, 65.9/40.1/26.7/18.2 (BP=0.967, ratio=0.968, syslen=66715, reflen=68951)
# last: Generate test with beam=4: BLEU4 = 32.52, 65.8/40.0/26.6/18.2 (BP=0.968, ratio=0.969, syslen=66805, reflen=68951)
