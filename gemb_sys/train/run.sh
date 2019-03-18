#!/bin/bash
method=$1
emb_model=asym
graph=$2
file=$3
python __main__.py --input ../../mydata/${graph}_edge.txt --graph-format edgelist --output emb/${graph}_${method}_${emb_model}.embedding --model-v ${method} --emb_model ${emb_model} --rw-file walks/deepwalk_${graph}.walks --window-size 5 --negative-ratio 1 --epochs 20 --degree-power 1 --pair-file ${file} --batch-size 500
#python __main__.py --input ../../mydata/blogcatalog_edge.txt --graph-format edgelist --output emb/${method}_${emb_model}.embedding --model-v ${method} --emb_model ${emb_model} --rw-file emb/blogcatalog.embeddings.walks.0 --window-size 5 --negative-ratio 1
