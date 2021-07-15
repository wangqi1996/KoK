export CUDA_VISIBLE_DEVICES=$1
export TOKENIZERS_PARALLELISM=false
export MKL_THREADING_LAYER=GUN
rm -rf $LOGDIR

fairseq-train /home/data_ti5_c/wangdq/data/fairseq/wmt14/ende-fairseq2 \
  --save-dir ~/save/wmt14/ \
  --arch tm_vaswani_wmt_en_de_big \
  --share-all-embeddings --ddp-backend=no_c10d \
  --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
  --lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07 \
  --dropout 0.3 --weight-decay 0.0 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --max-tokens 4000 \
  --update-freq 2 \
  --fp16 \
  --log-format simple --log-interval 100 \
  --save-interval-updates 500 --keep-best-checkpoints 5 --no-epoch-checkpoints --keep-interval-updates 5 \
  --max-update 300000 \
  --num-workers 0 \
  --eval-bleu \
  --eval-bleu-args '{"beam": 4, "max_len_a": 1.2, "max_len_b": 10}' \
  --eval-bleu-detok moses --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  -s de -t en
