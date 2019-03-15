#!/bin/bash
method=$1
graph=$2
emb_model=${3} #asym
python __main__.py --embedding-file ../train/emb/${graph}_${method}_${emb_model}.embedding --classification --clf-ratio 0.1 --label-file ../../mydata/email_label.txt
