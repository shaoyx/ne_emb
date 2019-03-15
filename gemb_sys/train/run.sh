#!/bin/bash
method=$1
emb_model=asym
python __main__.py --input ../../mydata/blogcatalog_edge.txt --graph-format edgelist --output emb/${method}_${emb_model}.embedding --model-v ${method} --emb_model ${emb_model} --rw-file emb/blogcatalog.embeddings.walks.0 --window-size 5 --negative-ratio 1
