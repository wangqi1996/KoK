
Code for our AAAI 2022 paper "Non-Parametric Online Learning from Human Feedback for Neural Machine Translation". 


This project implements our  KoK(KNN-Over-KNN) as well as Vanilla KNN-MT.
The implementation is build upon [Adaptive kNN-MT]( https://github.com/zhengxxn/adaptive-knn-mt.git), many thanks to the authors for making their code avaliable.

## Requirements and Installation

* pytorch version >= 1.5.0
* python version >= 3.6
* faiss-gpu >= 1.6.5
* pytorch_scatter = 2.0.5
* 1.19.0 <= numpy < 1.20.0

You can install this project by
```
pip install --editable ./
```

## Instructions

We use an example to show how to use our codes.

### Pre-trained Model and Data

The pre-trained translation model can be downloaded from [this site](https://github.com/pytorch/fairseq/blob/master/examples/wmt19/README.md).
We use the De->En Single Model for all experiments.

For convenience, We provide pre-processed [EMEA data](https://drive.google.com/file/d/17ACu2wJ-6Z2vVSu-R9YrTXQ3kINggKpi/view?usp=sharing) and [JRC-Acquis Dataset](https://drive.google.com/file/d/1hi0vjzdWx0FIS335GL2qXqcbdjy-lmXN/view?usp=sharing).


### Inference with KoK
```
export CUDA_VISIBLE_DEVICES=$1

DATA_PATH=''
MODEL_PATH=''
OUTPUT_PATH=''
split=train

python fairseq_cli/knn_generate.py $DATA_PATH \
  --gen-subset $split \
  --path $MODEL_PATH --arch knn_transformer_wmt19 \
  --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
  --scoring sacrebleu \
  --batch-size 1 \
  --tokenizer moses --remove-bpe \
  --results-path $OUTPUT_PATH \
  --model-overrides "{'k': 8, 'lambda_value': 0.3, 'temperature_value': 100, 'label_temperature_value': 100,
  'arch': 'knn_transformer_wmt19', 'knn_type': 'label-datastore', 'distance_threshold': 100
  }"
```
We recommend you to use the below command:
```
bash scripts/ema/all_kok.sh 1 c
```

### Inference with KNN-MT
We also provide scripts to do vanilla kNN-MT inference

```
DATA_PATH=''
MODEL_PATH=''
OUTPUT_PATH='
split=train

python fairseq_cli/knn_generate.py $DATA_PATH \
  --gen-subset $split \
  --path $MODEL_PATH --arch knn_transformer_wmt19 \
  --beam 4 --lenpen 0.6 --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
  --scoring sacrebleu \
  --batch-size 1 \
  --tokenizer moses --remove-bpe \
  --results-path $OUTPUT_PATH \
  --model-overrides "{'k': 8, 'lambda_value': 0.2, 'temperature_value': 10, 'arch': 'knn_transformer_wmt19', \
  'knn_type': 'normal' }"
```
We recommend you to use below command:
```
bash scripts/ema/all_knn.sh 1 c
```