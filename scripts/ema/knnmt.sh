export CUDA_VISIBLE_DEVICES=$1

DATA_PATH=/home/wangdq/data/split_docs/bin/$2/$3
PROJECT_PATH=/home/data_ti5_c/wangdq/code/knn-mt
MODEL_PATH=/home/data_ti5_c/wangdq/model/fairseq/pretrain/wmt19/deen/model/wmt19.de-en.ffn8192.pt
OUTPUT_PATH=/home/wangdq/output/$4
split=train

python $PROJECT_PATH/experimental_generate.py $DATA_PATH \
  --gen-subset $split \
  --path $MODEL_PATH --arch transformer_wmt19_de_en_with_datastore \
  --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
  --scoring sacrebleu \
  --batch-size 1 \
  --tokenizer moses --remove-bpe \
  --results-path $OUTPUT_PATH \
  --model-overrides "{'k': 8, 'knn_lambda_value': $5,
  'knn_temperature_value': 10, 'knn_type': 'normal'}"

tail -1 $OUTPUT_PATH/generate-$split.txt
