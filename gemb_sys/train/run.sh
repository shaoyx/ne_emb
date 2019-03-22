#!/bin/bash
method=$1
emb_model=asym
graph=$2
file=$3
fac=$4
echo "python __main__.py --input ../../mydata/${graph}_edge.txt --graph-format edgelist --output emb/${graph}_${method}_${emb_model}_${fac}.embedding --model-v ${method} --emb_model ${emb_model} --rw-file ${file} --window-size 10 --negative-ratio 5 --epochs 20 --degree-power 1 --pair-file ${file} --batch-size 1000 --epoch-fac ${fac}"
python __main__.py --input ../../mydata/${graph}_edge.txt --graph-format edgelist --output emb/${graph}_${method}_${emb_model}_${fac}.embedding --model-v ${method} --emb_model ${emb_model} --rw-file ${file} --window-size 10 --negative-ratio 5 --epochs 20 --degree-power 0 --pair-file ${file} --batch-size 1000 --epoch-fac ${fac} --app-sample ${fac}
#python __main__.py --input ../../mydata/blogcatalog_edge.txt --graph-format edgelist --output emb/${method}_${emb_model}.embedding --model-v ${method} --emb_model ${emb_model} --rw-file emb/blogcatalog.embeddings.walks.0 --window-size 5 --negative-ratio 1
