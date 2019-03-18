#!/bin/bash
method=$1
graph=$2
emb_model=${3} #asym
ratio=0.1
python __main__.py --embedding-file ../train/emb/${graph}_${method}_${emb_model}.embedding --classification --clf-ratio ${ratio} --label-file ../../mydata/${graph}_label.txt --input ../../mydata/${graph}_edge.txt
