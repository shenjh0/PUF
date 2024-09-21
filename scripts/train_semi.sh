#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# modify these augments if you want to try other datasets, splits or methods
# dataset: ['cdd', 'sysu','levir', 'levir_au']
# method: ['puf','supervised']
# split: ['1%', '5%', '10%', '20%']
# remember to modify the data_root in corresponding config
dataset='levir_au'
method='puf'
split='5%'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$split
mkdir -p $save_path

cp ${method}.py $save_path/${method}_$now.py

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    $method.py \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log