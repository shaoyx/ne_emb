# vcsample使用说明

Embedding training model

running on python3

## 命令行参数说明

### 输入输出相关

--input 输入图文件名，可以是邻接表或者边表，不可缺省

--output 输出embedding文件名；可选

--label-file 输入节点标签文件名；可选

--graph-format 输入图格式，可以是adjlist（邻接表，默认）或edgelist（边表）

--weighted 是否有权重，写了表示有权重，下同；带权图相关代码尚不完善

--directed 是否有向

--embedding-file 预训练的embedding结果，若设置该项，则除link prediction外都可跳过embedding训练，直接评测

### Embedding训练相关

--representation-size embedding的维度；默认128

--epochs 训练epoch数，默认20

--epoch-fac 一个epoch包含的节点数除以节点总数的值，默认50

--batch-size 默认1000

--lr 学习率，默认0.001

--negative-ratio 负采样数和正样例数的比值n。因为是按“一个正样例batch+n个负样例batch”轮流训练的，所以必须是非负整数；默认5

### 算法相关

--model-v 指定vertex采样方式，不可缺省

--model-c 指定context采样方式，若缺省则默认与model_v相同，相同的model只会做1次初始化

#### app

--app-jump-factor 停止的概率；默认0.15

--app-step 随机游走的最大步数；默认80

#### deepwalk

--window-size 采样context时，对应的窗格大小，默认10