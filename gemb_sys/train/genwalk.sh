#!/bin/bash
method=$1
graph=$2
graph_file=$3
python walker.py --input ${graph_file} --output walks/${method}_${graph}.walks --graph-format edgelist
