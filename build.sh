#!/bin/bash
data_dir="/home/wuliu/datasets/LibriSpeechMix/train-960"
data_path="train_data.list"
json_path="list/train-960-1mix.jsonl"


for((num_speakers = 1; num_speakers <= 3; num_speakers++));
do
    python utils/build_train.py --random_seed 3407 \
                            --data_dir $data_dir \
                            --data_path $data_path \
                            --num_speakers $num_speakers \
                            --sampling_rate 16000 \
                            --json_path $json_path &
done
