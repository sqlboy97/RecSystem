import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))
from online import stream_sc, SIMILAR_DS, pool, HOT_DS, ARTICLE_DS
from datetime import datetime
from setting.default import DefaultConfig
import redis
import json
import time
import setting.logging as lg
import logging


logger = logging.getLogger('online')


class OnlineRecall(object):
    """在线处理计算平台
    """

    def __init__(self):
        self.client = redis.StrictRedis(host=DefaultConfig.REDIS_HOST,
                                        port=DefaultConfig.REDIS_PORT,
                                        db=10)

        self.k = 10

    def _update_hot_redis(self):
        """
        在线实时收集用户日志中的文章，进行热门更新
        :return:
        """
        # 定义全局结果
        client = self.client

        def hot_function(rdd):
            # rdd当中包含有个行为数据
            # # echo {\"actionTime\":\"2019-04-10 21:04:39\",\"readTime\":\"\",
            #         # \"channelId\":18,\"param\":{\"action\": \"click\", \"userId\": \"2\"
            #         # , \"articleId\": \"14299\", \"algorithmCombine\": \"C2\"}} >> userClick.log
            for row in rdd.collect():
                if row['param']['action'] == 'exposure' or row['param']['action'] == 'read':
                    pass
                else:
                    logger.info("{} INFO: store channel_id:{} add hot article:{}".format(
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), row['channelId'], row["param"]["articleId"]))

                    client.zincrby('ch:{}:hot'.format(row['channelId']), 1, row['param']['articleId'])

        HOT_DS.map(lambda x: json.loads(x[1])).foreachRDD(hot_function)
        return None

    def _update_new_redis(self):
        """
        在线实时收集后台传来的新文章 channel_id,article_id
        :return:
        """
        # 定义全局结果
        client = self.client

        def add_new_article(rdd):

            for row in rdd.collect():
                channel_id, article_id = row.split(',')
                logger.info("{} INFO: store channel_id:{} add new article:{}".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), channel_id, article_id))
                client.zadd('ch:{}:new'.format(channel_id), {article_id: time.time()})

        ARTICLE_DS.map(lambda x: x[1]).foreachRDD(add_new_article)
        return None

    def _update_online_cb(self):
        """
        在线基于内容召回实时行为相似文章推荐
        :return:
        """
        # echo {\"actionTime\":\"2019-04-10 21:04:39\",\"readTime\":\"\",
        # \"channelId\":18,\"param\":{\"action\": \"click\", \"userId\": \"2\"
        # , \"articleId\": \"14299\", \"algorithmCombine\": \"C2\"}} >> userClick.log
        # 定义处理拿到数据的函数
        def foreachfunc(rdd):

            # 对RDD解析
            # 一条点击日志，十条点击日志
            for data in rdd.collect():
                # 读取字典中的数据，判断用户行为，如果点击、收藏、分享行为，获取hbase该文章相似的文章集合
                if data['param']['action'] in ["click", "collect", "share"]:
                    logger.info("{} INFO: get user clicked data:{}".format(
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data["param"]["articleId"]))
                    # 获取用户发生行为的文章
                    with pool.connection() as conn:
                        # 获取similar表

                        try:
                            sim_table = conn.table('article_similar')

                            _dic = sim_table.row(str(data["param"]["articleId"]).encode(), columns=[b"similar"])

                            _srt = sorted(_dic.items(), key=lambda obj: obj[1], reverse=True)  # 按相似度排序
                            if _srt:
                                topKSimIds = [int(i[0].split(b":")[1]) for i in _srt[:self.k]]

                                # 读取历史推荐结果，过滤推荐过的文章，新推召回荐结果写入cb_recall当中，写入历史表中

                                # 根据历史推荐集过滤，已经给用户推荐过的文章
                                history_table = conn.table("history_recall")

                                _history_data = history_table.cells(
                                    b"reco:his:%s" % data["param"]["userId"].encode(),
                                    b"channel:%d" % data["channelId"]
                                )
                                # print("_history_data: ", _history_data)

                                history = []
                                if len(data) >= 2:
                                    for l in data[:-1]:
                                        history.extend(eval(l))
                                else:
                                    history = []

                                # 根据历史召回记录，过滤召回结果
                                recall_list = list(set(topKSimIds) - set(history_data))

                                # print("recall_list: ", recall_list)
                                logger.info("{} INFO: store user:{} cb_recall data".format(
                                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), data["param"]["userId"]))
                                if recall_list:
                                    # 如果有推荐结果集，那么将数据添加到cb_recall表中，同时记录到历史记录表中
                                    logger.info(
                                        "{} INFO: get online-recall data".format(
                                            datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                                    recall_table = conn.table("cb_recall")

                                    recall_table.put(
                                        b"recall:user:%s" % data["param"]["userId"].encode(),
                                        {b"online:%d" % data["channelId"]: str(recall_list).encode()}
                                    )

                                    history_table.put(
                                        b"reco:his:%s" % data["param"]["userId"].encode(),
                                        {b"channel:%d" % data["channelId"]: str(recall_list).encode()}
                                    )
                        except Exception as e:
                            logger.info("{}, WARN: {}".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))

                        conn.close()
        # x为Kafka拿到的数据
        SIMILAR_DS.map(lambda x: json.loads(x[1])).foreachRDD(foreachfunc)


if __name__ == '__main__':
    lg.create_logger()
    # 开启spark streamin
    op = OnlineRecall()
    op._update_online_cb()
    op._update_hot_redis()
    op._update_new_redis()
    # 启动sreatming
    stream_sc.start()

    # 使用 ctrl+c 可以退出服务
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)
