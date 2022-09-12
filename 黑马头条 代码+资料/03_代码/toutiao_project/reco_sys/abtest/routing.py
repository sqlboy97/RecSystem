import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR))
print(BASE_DIR)
from concurrent import futures
from abtest import user_reco_pb2
from abtest import user_reco_pb2_grpc
from setting.default import DefaultConfig, RAParam
from server.reco_cent import RecoCenter
import hashlib
import json
import time
import grpc


def feed_recommend(user_id, channel_id, article_num, time_stamp):
    """
    1、根据用户ID来进行分流
    2、分配不同算法，以便在推荐中心进行相应算法排序和召回
    :param user_id:
    :param channel_id:
    :param article_num:
    :param time_stamp:
    :return:
    """
    # 1、封装参数,进行推荐使用，埋入日志
    class Temp(object):
        user_id = -10
        channel_id = -10
        article_num = -10
        time_stamp = -10
        algo = ''

    tp = Temp()
    tp.user_id = user_id
    tp.channel_id = channel_id
    tp.article_num = article_num
    tp.time_stamp = time_stamp

    # 2、根据用户ID进行相应判断，加密，添加用户的算法组合是什么
    if tp.user_id == "":
        tp.algo = ''
        return add_track([], tp)

    # 进行加密
    bucket = hashlib.md5(tp.user_id.encode()).hexdigest()[:1]
    if bucket in RAParam.BYPASS[0]['Bucket']:
        tp.algo = RAParam.BYPASS[0]['Strategy']
    else:
        tp.algo = RAParam.BYPASS[1]['Strategy']

    # 进行推荐中获取推荐结果
    # track = add_track([1, 2, 3, 4], tp)
    track = RecoCenter().feed_recommend_logic(tp)

    return track


# 定义服务端的类
class UserRecommendServicer(user_reco_pb2_grpc.UserRecommendServicer):
    """黑马推荐grpc服务类
    """

    def user_recommend(self, request, context):
        """
        feed流推荐逻辑
        :param request:
        :param context:
        :return:
        """

        # 1、接受参数处理
        user_id = request.user_id
        channel_id = request.channel_id
        article_num = request.article_num
        time_stamp = request.time_stamp

        # 2、根据参数推荐相应文章,返回推荐空结果
        # 埋点参数参考：
        # {
        #     "param": '{"action": "exposure", "userId": 1, "articleId": [1,2,3,4],  "algorithmCombine": "c1"}',
        #     "recommends": [
        #         {"article_id": 1, "param": {"click": "{"action": "click", "userId": "1", "articleId": 1, "algorithmCombine": 'c1'}", "collect": "", "share": "","read":""}},
        #         {"article_id": 2, "param": {"click": "", "collect": "", "share": "", "read":""}},
        #         {"article_id": 3, "param": {"click": "", "collect": "", "share": "", "read":""}},
        #         {"article_id": 4, "param": {"click": "", "collect": "", "share": "", "read":""}}
        #     ]
        #     "timestamp": 1546391572
        # }
        # test
        # class Temp(object):
        #     user_id = '1115629498121846784'
        #     algo = 'test'
        #     time_stamp = int(time.time() * 1000)
        #
        # _track = add_track([1, 2, 3, 4], Temp())
        # 经过分流的推荐
        _track = feed_recommend(user_id, channel_id, article_num, time_stamp)

        # 3、推荐结果进行协议封装
        # proto协议封装
        # 封装第二个推荐的参数
        # [param1(article_id, param2), param1(article_id, param2)..]
        _recommend = []
        for _ in _track['recommends']:

            _param2 = user_reco_pb2.param2(click=_['param']['click'],
                                           collect=_['param']['collect'],
                                           share=_['param']['share'],
                                           read=_['param']['read'])

            _p = user_reco_pb2.param1(article_id=_['article_id'], params=_param2)
            _recommend.append(_p)

        return user_reco_pb2.Track(exposure=_track['param'], recommends=_recommend, time_stamp=_track['timestamp'])


def serve():
    import setting.logging as lg
    lg.create_logger()
    # 启动服务器
    # 多线程服务器
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # 注册本地服务
    user_reco_pb2_grpc.add_UserRecommendServicer_to_server(UserRecommendServicer(), server)
    # 监听端口
    server.add_insecure_port(DefaultConfig.RPC_SERVER)

    # 开始接收请求进行服务
    server.start()
    # 使用 ctrl+c 可以退出服务
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)


if __name__ == '__main__':
    serve()

