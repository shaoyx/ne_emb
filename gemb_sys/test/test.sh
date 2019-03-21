#!/bin/bash
echo "method graph emb_model epoch task fac" 
method=$1
graph=$2
emb_model=${3} #asym
epoch=${4}
benchmark=${5} #--classification
fac=${6}
echo "python __main__.py --embedding-file ../train/emb/${graph}_${method}_${emb_model}${fac}.embedding${epoch} --${benchmark} --label-file ../../mydata/${graph}_label.txt --input ../../mydata/${graph}_edge.txt"
python __main__.py --embedding-file ../train/emb/${graph}_${method}_${emb_model}${fac}.embedding${epoch} --${benchmark} --label-file ../../mydata/${graph}_label.txt --input ../../mydata/${graph}_edge.txt
