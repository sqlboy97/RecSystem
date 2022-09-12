"""
编写各种更新任务的函数
"""
from offline.update_article import UpdateArticle
from offline.update_user import UpdateUserProfile
from offline.update_recall import UpdateRecall
from offline.update_ctrfeature import FeaturePlatform


def update_article_profile():
    """文章画像的定时更新逻辑
    :return:
    """
    ua = UpdateArticle()
    sentence_df = ua.merge_article_data()
    if sentence_df.rdd.collect():
        rank, idf = ua.generate_article_label(sentence_df)
        articleProfile = ua.get_article_profile(rank, idf)
        ua.compute_article_similar(articleProfile)


def update_user_profile():
    """用户画像定时更新逻辑
    :return:
    """
    op = UpdateUserProfile()
    if op.update_user_action_basic():
        op.update_user_label()
        op.update_user_info()


def update_recall():
    """
    更新用户的召回集
    :return:
    """
    # 200就是ALS模型的计算推荐的文章数量
    udp = UpdateRecall(200)
    udp.update_als_recall()
    udp.update_content_recall()


def update_ctr_feature():
    """
    定时更新用户、文章特征
    :return:
    """
    fp = FeaturePlatform()
    fp.update_user_ctr_feature_to_hbase()
    fp.update_article_ctr_feature_to_hbase()