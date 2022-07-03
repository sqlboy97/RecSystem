# 总览

DeepCTR的整体结构如下图所示：

<img src="DeepCTR.assets/DeepCTR.png" style="zoom: 80%;" />

主要包括四个模块（module）：

- contirb：包含一些RNN层。（不太常用）
- layers：常用的网络层。
- models：常见的推荐模型，包括一些常见的多任务（multitask）模型和序列（sequence）模型。
- estimator：高层封装API，可以用来直接构造网络。

还包括三个.py文件：

- feature_column.py：处理特征的一些类和函数。
- inputs.py：处理输入的一些工具函数。
- utils.py：包含一些工具类。

# feature_column.py

文件中包含了DeepCTR中定义的三个特征类型：

- 单值离散型特征（class SparseFeat）
- 多值离散型特征（class VarLenSparseFeat）
- 数值型特征（class DenseFeat）

还有四个用来处理特征的函数：

- `def build_input_features`
- `def get_feature_names`
- `def input_from_feature_columns`
- `def get_linear_logit`

## 数据类别



### class SparseFeat

单值离散型特征

```python
class SparseFeat(namedtuple('SparseFeat',
                            ['name', 'vocabulary_size', 'embedding_dim', 'use_hash', 'vocabulary_path', 'dtype', 'embeddings_initializer',
                             'embedding_name',
                             'group_name', 'trainable'])):
    __slots__ = ()

    def __new__(cls, name, vocabulary_size, embedding_dim=4, use_hash=False, vocabulary_path=None, dtype="int32", embeddings_initializer=None,
                embedding_name=None,
                group_name=DEFAULT_GROUP_NAME, trainable=True):

        if embedding_dim == "auto":  # 如果是自动获取embedding的维度，则根据下面的原则设定
            embedding_dim = 6 * int(pow(vocabulary_size, 0.25))
        if embeddings_initializer is None:  # 未指定embedding的初始化，则指定一个
            embeddings_initializer = RandomNormal(mean=0.0, stddev=0.0001, seed=2020)

        if embedding_name is None:
            embedding_name = name

        return super(SparseFeat, cls).__new__(cls, name, vocabulary_size, embedding_dim, use_hash, vocabulary_path, dtype,
                                              embeddings_initializer,
                                              embedding_name, group_name, trainable)

    def __hash__(self):
        return self.name.__hash__()
```

### class VarLenSparseFeat

多值离散型特征

```python
class VarLenSparseFeat(namedtuple('VarLenSparseFeat',
                                  ['sparsefeat', 'maxlen', 'combiner', 'length_name', 'weight_name', 'weight_norm'])):
    __slots__ = ()

    def __new__(cls, sparsefeat, maxlen, combiner="mean", length_name=None, weight_name=None, weight_norm=True):
        return super(VarLenSparseFeat, cls).__new__(cls, sparsefeat, maxlen, combiner, length_name, weight_name,
                                                    weight_norm)

    @property  # 用@property装饰器创建只读属性，方法可以像属性一样被访问
    def name(self):
        return self.sparsefeat.name

    @property
    def vocabulary_size(self):
        return self.sparsefeat.vocabulary_size

    @property
    def embedding_dim(self):
        return self.sparsefeat.embedding_dim

    @property
    def use_hash(self):
        return self.sparsefeat.use_hash

    @property
    def vocabulary_path(self):
        return self.sparsefeat.vocabulary_path

    @property
    def dtype(self):
        return self.sparsefeat.dtype

    @property
    def embeddings_initializer(self):
        return self.sparsefeat.embeddings_initializer

    @property
    def embedding_name(self):
        return self.sparsefeat.embedding_name

    @property
    def group_name(self):
        return self.sparsefeat.group_name

    @property
    def trainable(self):
        return self.sparsefeat.trainable

    def __hash__(self):
        return self.name.__hash__()
```

### class DenseFeat

数值型特征

```python

    def __hash__(self):
        return self.name.__hash__()
```

## 函数

### def build_input_features

```python
```

