from pyspark import SparkConf
from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
from setting.default import DefaultConfig

# happybase
import happybase

#  用于读取hbase缓存结果配置
pool = happybase.ConnectionPool(size=10, host='hadoop-master', port=9090)


# 1、创建spark streaming的配置
conf = SparkConf()

conf.setAll(DefaultConfig.SPARK_ONLINE_CONFIG)
sc = SparkContext(conf=conf)
# 用来启动streaming
stream_sc = StreamingContext(sc, 60)

# 2、streamin 对接Kafka的配置
similar_kafkaParams = {"metadata.broker.list": DefaultConfig.KAFKA_SERVER, "group.id": 'similar'}
SIMILAR_DS = KafkaUtils.createDirectStream(stream_sc, ['click-trace'], similar_kafkaParams)

# 3、创建HOT和NEW Kafka配置
kafkaParams = {"metadata.broker.list": DefaultConfig.KAFKA_SERVER}
HOT_DS = KafkaUtils.createDirectStream(stream_sc, ['click-trace'], kafkaParams)

# new article的topic new-article
# 后台和推荐对接
ARTICLE_DS = KafkaUtils.createDirectStream(stream_sc, ['new-article'], kafkaParams)



