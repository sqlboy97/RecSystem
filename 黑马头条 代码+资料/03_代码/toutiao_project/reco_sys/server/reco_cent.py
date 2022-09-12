import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))
import hashlib
from setting.default import RAParam
from server.utils import HBaseUtils
from server import pool
from server import recall_server
from datetime import datetime
from server.redis_hbase_cache import get_cache_from_redis_hbase
from server.sort_service import lr_sort_service, lrftrl_sort_service
import logging
import json


sort_dict = {
    "LR": lr_sort_service,
    "FTRL": lrftrl_sort_service
}

logger = logging.getLogger('recommend')


def add_track(res, temp):
    """
    封装埋点参数
    :param res: 推荐文章id列表
    :param cb: 合并参数
    :param rpc_param: rpc参数
    :return: 埋点参数
        文章列表参数
        单文章参数
    """
    # 添加埋点参数
    track = {}

    # 准备曝光参数
    # 全部字符串形式提供，在hive端不会解析问题
    _exposure = {"action": "exposure", "userId": temp.user_id, "articleId": json.dumps(res),
                 "algorithmCombine": temp.algo}

    track['param'] = json.dumps(_exposure)
    track['recommends'] = []

    # 准备其它点击参数
    for _id in res:
        # 构造字典
        _dic = {}
        _dic['article_id'] = _id
        _dic['param'] = {}

        # 准备click参数
        _p = {"action": "click", "userId": temp.user_id, "articleId": str(_id),
              "algorithmCombine": temp.algo}

        _dic['param']['click'] = json.dumps(_p)
        # 准备collect参数
        _p["action"] = 'collect'
        _dic['param']['collect'] = json.dumps(_p)
        # 准备share参数
        _p["action"] = 'share'
        _dic['param']['share'] = json.dumps(_p)
        # 准备detentionTime参数
        _p["action"] = 'read'
        _dic['param']['read'] = json.dumps(_p)

        track['recommends'].append(_dic)

    track['timestamp'] = temp.time_stamp
    return track


class RecoCenter(object):
    """推荐中心
    """
    def __init__(self):
        self.hbu = HBaseUtils(pool)
        self.recall_service = recall_server.ReadRecall()

    def feed_recommend_logic(self, temp):
        """
        推荐中心的逻辑
        :param temp:
        :return:
        """
        # 根据请求时间戳判断
        try:
            last_stamp = self.hbu.get_table_row('history_recommend',
                                                'reco:his:{}'.format(temp.user_id).encode(),
                                                'channel:{}'.format(temp.channel_id).encode(),
                                                include_timestamp=True)[1]
            logger.info("{} INFO get user_id:{} channel:{} history last_stamp".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))
        except Exception as e:
            logger.warning("{} WARN read history recommend exception:{}".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
            last_stamp = 0

        # 1、如果该用户某频道的历史推荐记录中的最近时间戳 < temp.time_stamp，下拉刷新
        # 获取历史最近时间戳
        # 1558143073173,
        if last_stamp < temp.time_stamp:
            # 测试
            # 返回召回结果
            # 返回前面一个历史记录时间戳
            # track = add_track([44657, 14961, 17522, 43894, 44412, 16000, 14208, 44419, 17802, 14223, 18836], temp)
            # 1、加入缓存逻辑
            res = get_cache_from_redis_hbase(temp, self.hbu)
            if not res:
                logger.info("{} INFO redis is null get user_id:{} channel:{} recall/sort data".
                            format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))
                res = self.user_reco_list(temp)

            # 2、没有加入缓存的测试
            # res = self.user_reco_list(temp)

            temp.time_stamp = int(last_stamp)
            track = add_track(res, temp)

            return track
        else:
            # 2、如果该用户某频道的历史推荐记录中的最近时间戳 >= temp.time_stamp 上滑获取历史记录
            # 读取历史记录，cells
            # 获取所有版本的数据
            # timestamp=temp.time_stamp + 1, 包含自己请求的时间戳数据
            logger.info("{} INFO read user_id:{} channel:{} history recommend data".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))
            try:
                row = self.hbu.get_table_cells('history_recommend',
                                               'reco:his:{}'.format(temp.user_id).encode(),
                                               'channel:{}'.format(temp.channel_id).encode(),
                                               timestamp=temp.time_stamp + 1,
                                               include_timestamp=True)
            except Exception as e:
                logger.warning("{} WARN read history recommend exception:{}".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), e))
                row = []
                res = []

            # 进行逻辑判断
            # 1、如果没有历史数据，返回时间戳0以及结果空列表
            # 2、如果历史数据只有一条，返回这一条历史数据以及时间戳正好为请求时间戳，修改时间戳为0，表示后面请求以后就没有历史数据了(APP的行为就是翻历史记录停止了)
            # 3、如果历史数据多条，返回最近的第一条历史数据，然后返回之后第二条历史数据的时间戳
            if not row:
                # 表示看历史记录到头了一个标志
                temp.time_stamp = 0
                # 返回空结果
                res = []
            elif len(row) == 1 and row[0][1] == temp.time_stamp:
                # 123456789102 最后一条历史记录
                res = eval(row[0][0])
                temp.time_stamp = 0
            elif len(row) >= 2:
                # 5,4, 3,2,1
                res = eval(row[0][0])
                temp.time_stamp = int(row[1][1])

            # 最终结果封装
            res = list(map(int, res))

            track = add_track(res, temp)
            # 因为获取历史，不是新请求召回排序刷新
            track['param'] = ''
            return track

    def user_reco_list(self, temp):
        """
        用户的推荐获取（召回结果获取+过滤+排序 + 返回推荐）
        :param temp:
        :return:
        """
        # 所有合并的结果
        reco_set = []
        # -  1、循环算法组合参数，遍历不同召回结果合并，18
        for _num in RAParam.COMBINE[temp.algo][1]:
            if _num == 103:
                # 读取新文章的结果，temp.channel_id
                _res = self.recall_service.read_redis_new_article(temp.channel_id)
                reco_set = list(set(reco_set).union(set(_res)))
            elif _num == 104:
                # 读取热门文章的数据
                _res = self.recall_service.read_redis_hot_article(temp.channel_id)
                reco_set = list(set(reco_set).union(set(_res)))
            else:
                # 读取具体编号的对应表cb_recall,对应召回算法的召回结果als, content, online
                _res = self.recall_service.\
                    read_hbase_recall_data(RAParam.RECALL[_num][0],
                                           'recall:user:{}'.format(temp.user_id).encode(),
                                           '{}:{}'.format(RAParam.RECALL[_num][1], temp.channel_id).encode())
                reco_set = list(set(reco_set).union(set(_res)))

        # - 2、过滤，该请求频道(18)的历史推荐记录过滤），推荐频道0频道
        #   - 0：APP  推荐(循环所有的频道召回结果)，0 频道也有历史记录
        # temp.channel_id频道这个用户历史记录进行过滤
        history_list = []

        try:
            data = self.hbu.get_table_cells('history_recommend',
                                            'reco:his:{}'.format(temp.user_id).encode(),
                                            'channel:{}'.format(temp.channel_id).encode())
            for _ in data:
                history_list = list(set(history_list).union(set(eval(_))))

            logger.info("{} INFO read user_id:{} channel_id:{} history data".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))

        except Exception as e:
            # 打印日志
            logger.warning(
                "{} WARN filter history article exception:{}".format(datetime.now().
                                                                     strftime('%Y-%m-%d %H:%M:%S'), e))
        # 获取0频道的结果
        try:
            data = self.hbu.get_table_cells('history_recommend',
                                            'reco:his:{}'.format(temp.user_id).encode(),
                                            'channel:{}'.format(0).encode())
            for _ in data:
                history_list = list(set(history_list).union(set(eval(_))))
            logger.info("{} INFO filter user_id:{} channel:{} history data".format(
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, 0))

        except Exception as e:
            logger.warning(
                "{} WARN filter history article exception:{}".format(datetime.now().
                                                                     strftime('%Y-%m-%d %H:%M:%S'), e))

        reco_set = list(set(reco_set).difference(set(history_list)))

        # 过滤
        # - 3、对结果，排序
        # 排序代码逻辑 reco_set article_id
        # _sort_num = RAParam.COMBINE[temp.algo][2][0]
        # reco_set = sort_dict[RAParam.SORT[_sort_num]](reco_set, temp, self.hbu)
        if reco_set:
            _sort_num = RAParam.COMBINE[temp.algo][2][0]
            reco_set = sort_dict[RAParam.SORT[_sort_num]](reco_set, temp, self.hbu)
        # reco_set排序之后的推荐结果（文章列表）

        # - 4、返回结果：
        if not reco_set:
            return reco_set
        else:
            # - 如果有数据，小于需要推荐文章的数量N之后，放入历史推荐记录中history_recommend，返回结果给用
            if len(reco_set) <= temp.article_num:
                res = reco_set
            else:
                #   - 如果有，350篇，取出N个，进行返回推荐，放入历史记录history_recommend
                #     - (350- N)个文章，放入wait_recommend
                res = reco_set[:temp.article_num]

                logger.info(
                    "{} INFO put user_id:{} channel:{} wait data".format(
                        datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))

                # 放入剩下多余的数据到wait_recommend当中
                self.hbu.get_table_put('wait_recommend',
                                       'reco:{}'.format(temp.user_id).encode(),
                                       'channel:{}'.format(temp.channel_id).encode(),
                                       str(reco_set[temp.article_num:]).encode(),
                                       timestamp=temp.time_stamp)

            # 放入历史记录
            self.hbu.get_table_put('history_recommend',
                                   'reco:his:{}'.format(temp.user_id).encode(),
                                   'channel:{}'.format(temp.channel_id).encode(),
                                   str(res).encode(),
                                   timestamp=temp.time_stamp)
            # 放入历史记录日志
            logger.info(
                "{} INFO store recall/sorted user_id:{} channel:{} history_recommend data".format(
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'), temp.user_id, temp.channel_id))

            return res








