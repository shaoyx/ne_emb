#!/bin/bash
echo "deepwalk blogcatalog asym 0.1 _5 classification"
method=$1
graph=$2
emb_model=${3} #asym
ratio=${4}
epoch=${5}
benchmark=${6} #--classification
echo "python __main__.py --embedding-file ../train/emb/${graph}_${method}_${emb_model}.embedding${epoch} --${benchmark} --clf-ratio ${ratio} --label-file ../../mydata/${graph}_label.txt --input ../../mydata/${graph}_edge.txt"
python __main__.py --embedding-file ../train/emb/${graph}_${method}_${emb_model}.embedding${epoch} --${benchmark} --clf-ratio ${ratio} --label-file ../../mydata/${graph}_label.txt --input ../../mydata/${graph}_edge.txt
