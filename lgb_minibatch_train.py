import lightgbm as lgb
from utils import IterMiniBatches
from sklearn.datasets import train_test_split

# 第一步, 初始化模型为None, 设置模型参数
GBM = None

PARAMS = {
    "task": "train",
    "application": "regression", # 目标函数
    "boosting_type": "gbdt", # 设置提升类型
    "learning_rate": 0.01, # 学习速率
    "num_leaves": 50, # 叶子节点数
    "tree_learner": "serial",
    "min_data_in_leaf": 100,
    "metric": ["l1", "l2", "rmse"], # l1:mae, 12:mse # 评估函数
    "max_bin": 255,
    "num_trees": 300
}

# 第二步, 流式读取数据(每次10w)
minibatch_train_iterators = IterMiniBatches(minibatch_size=100000)

for i, (X_, y_) in enumerate(minibatch_train_iterators):

    # 创建lgb数据集
    X_train, Y_test, y_train, y_test = train_test_split(X_, y_, test_size=0.1, random_state=0)

    y_train = y_train.ravel()

    lgb_train = lgb.Dataset(X_train, y_train)

    lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

    # 第三步, 增量训练模型
    # 重点!!! 通过 init_model 和 keep_training_booster 两个参数实现增量训练
    model = lgb.train(
        params=PARAMS,
        train_set=lgb_train,
        num_boost_round=1000,
        valid_sets=lgb_eval
        init_model=GBM,  # 如果gbm不为None, 那么就是在上次的基础上接着训练
#        feature_name=x_cols,
        early_stopping_rounds=10,
        verbose_eval=False,
        keep_training_booster=True  # 增量训练
    )

    print(f"{i} time")

    score_train = dict([(s[1], s[2]) for s in model.eval_train()])
    print("当前模型在训练集得分是: mae=%.4f, mse=%.4f,  rmse=%.4f" % (score_train["l1"], score_train["l2"], score_train["rmse"]))



# https://blog.csdn.net/lizz2276/article/details/125929215