https://www.jianshu.com/p/d65157c51082
https://www.jianshu.com/p/662cca93d315

在一个app中，通常需要对多个业务domain的商品进行ctr预估，比如在jj场景中，存在多个业务domain，例如：jjpdy，cfpdy，都需要进行ctr预估。

对于这种存在多个domain的场景，通常的做法是针对每个domain使用自己独有的数据训练各自的模型，并单独部署上线。
这种做法存在一些缺点，首先，一些domain的数据量比较小，数据比较稀疏，可能一个月只有几w的正样本，模型可能难以得到充分的训练，没有办法拟合线上的真实分布；其次，针对每个业务线都单独训练，部署一套作业，非常浪费计算资源与人力资源。

不同domian的用户和商品会有相当大一部分的交集，比如在cf这个场景下，真正会产生浏览点击的用户，大多都是对jj感兴趣的用户，且商品大多也是jj，因此不同domain之间的信息共享，一定程度上可以提升ctr预估模型的效果。
但是同时，不同的domain的用户行为存在一定的差异，会导致数据分布存在一定差异，简单地混合所有domain的数据来学习一个共享的模型，用于所有domain的ctr预估，可能达不到理想效果。

一 One Model to Serve All: Star Topology Adaptive Recommender for Multi-Domain CTR Prediction
论文中提出的方法称作star topology adaptive recom- mender (STAR)，其整体的结构如下图所示：

该结构主要包含三个主要的模块，分别是：partitioned normalization (PN)、star topology fully-connected neural network (star topology FCN)和auxiliary network，接下来，我们对这三部分进行分别介绍。


