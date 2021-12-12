export CUDA_VISIBLE_DEVICES=$1

DATA_PATH=/home/data_ti5_c/wangdq/data/ema/data/split_docs/bin/$2/$3
MODEL_PATH=/home/data_ti5_c/wangdq/model/fairseq/pretrain/wmt19/deen/model/wmt19.de-en.ffn8192.pt
OUTPUT_PATH=/home/wangdq/output/$4
split=train

python fairseq_cli/knn_generate.py $DATA_PATH \
  --gen-subset $split \
  --path $MODEL_PATH --arch knn_transformer_wmt19 \
  --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
  --scoring sacrebleu \
  --batch-size 1 \
  --tokenizer moses --remove-bpe \
  --results-path $OUTPUT_PATH \
  --model-overrides "{'k': 8, 'lambda_value': 0.2, 'temperature_value': 10,
  'arch': 'knn_transformer_wmt19', 'knn_type': 'normal'}"

tail -1 $OUTPUT_PATH/generate-$split.txt
