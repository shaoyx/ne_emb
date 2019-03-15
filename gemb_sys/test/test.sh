#!/bin/bash
method=$1
emb_model=asym
python __main__.py --embedding-file ../train/emb/${method}_${emb_model}.embedding --classification --clf-ratio 0.1 --label-file ../../mydata/blogcatalog_label.txt
