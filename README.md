# src
ne&amp;sample
### Changes to OpenNE
增加了MH-Walk

修改__main__.py使得LINE不会在每个epoch之后都做test，只在最后做一次

增加了lpWalk

#### 新增 & 问题

增加了APP

其中，默认iters=200, sample=200，按原代码实际的采样个数是node_num x iters x sample，batch_size = 1, epoch = 1, 时间复杂度非常大。

一方面，仅仅是采样的复杂度就已经远大于其他算法了，比如deepwalk是node_num x num_paths x walk_length，而它的num_paths和walk_length都比较小(10, 80)，所以我想在APP里面也把参数调小一点。

另一方面，原代码每一步都做更新，但是在点数很多的时候，会导致非常大的更新次数。所以，除非不用tensorflow做训练，否则大概很难有效地复现APP算法。不知调大batch_size对结果会有怎样的影响。我试过batch_size=1000, epoch=10和batch_size=100, epoch=1，结果差不多

当前参数：iters=10, sample=50, batch_size=1000, epoch=10, learning_rate=0.001(adam) 其他参数按原代码：停止概率jump_factor=0.15, 最大步数step=10, negative_ratio=5

在email上的表现：micro F1 = 0.6612326043737574, macro F1 = 0.37598095720575286 比deepwalk差不到0.1；时间和line差不多。可能需要调参

### nesampler
修改了__main__.py, line.py, node2vec.py，增加了trainer.py

trainer是根据line的二阶部分修改而成的，根据权重建立alias表，生成负采样表，做mini-batch梯度下降

line只保留了二阶部分，直接调用trainer

node2vec包含了deepwalk和node2vec算法，先调用walker（未改动）采样路径，再用myparser函数生成samples，最后调用trainer。

问题1：myparser有两种方式：

1. 每个点对都按权重1.0加入samples

2. 统计点对出现次数作为权重

不确定哪一种方式更好。当前采用第一种

问题2：细节问题，点对是否可以包含(v, v)这样自己到自己的点对？当前没有包含。

问题3：使用trainer，在email数据集上表现和原来差不多（不论parser是哪种方式），但是在cora和blogcatalog上表现很差。

尝试1：修改optimizer。原先是adam，考虑改成SGD，结果在email上都不收敛了（loss基本不变，分类效果非常差）

尝试2：调整学习率。没找到更好的学习率。

尝试3：取消batch的随机化，调整negative_ratio。似乎影响不大

尝试4：修改batch_size。没有提升cora上的效果。

尝试5：修改epoch数。即使在loss基本稳定的情况下，也基本没有提升cora上的效果。

修改前的超参数可参考line0.py（或者openne里的line.py）

尝试6：怀疑log_sigmoid出现NAN，做clip_by_value使得log的参数在1e-8~1.0的范围（原本是0.0~1.0）。结果没有明显变化。

尝试7：修改loss为nce_loss……结果很差。

### Other files
myresult是存放embedding结果和其他输出信息（info）的

mydata里面是数据集，包括edge, label, degreeRank(度数排序), info(基本信息), partinfo(连通性)，以及相关代码

mytest2是存放5次取平均的测试结果的

mycla是5次取平均的代码，含度数分段

combiner是把node2vec的grid search结果整合到一起的代码

