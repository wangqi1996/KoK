device_id=0
bs=4
lpv=0.6
temp=10
split=train

data_path=/path/to/data
model_path=/path/to/model
run_path=/path/to/code
export PYTHONPATH=$PYTHONPATH:$run_path
pip install --editable $run_path

# kok 
echo "KoK"
for doc in 50 100 200 500 1000;
do 
    file_list=$(ls -d $data_path/$doc/*/)
    for f in $file_list; 
    do  
        echo $f
        output_path=$f/output
        mkdir -p $output_path
        CUDA_VISIBLE_DEVICES=$device_id python3 $run_path/fairseq_cli/knn_generate.py $f \
            --gen-subset $split \
            --path $model_path --arch knn_transformer_wmt19 \
            --beam $bs --lenpen $lpv --max-len-a 1.2 --max-len-b 10 --source-lang de --target-lang en \
            --scoring sacrebleu \
            --batch-size 1 \
            --tokenizer moses --remove-bpe \
            --results-path $output_path \
            --model-overrides "{'k': 8, 'lambda_value': 0.3, 'temperature_value': $temp, 'label_temperature_value': $temp,
            'arch': 'knn_transformer_wmt19', 'knn_type': 'label-datastore', 'distance_threshold': 100
            }"
    done 
done 

