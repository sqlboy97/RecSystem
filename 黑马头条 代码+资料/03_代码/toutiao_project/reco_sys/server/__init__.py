import happybase
import redis
from setting.default import DefaultConfig

# hbase连接池
poolll = happybase.ConnectionPool(size=10, host='192.168.19.137', port=9090)

# redis连接
redis_client = redis.StrictRedis(host=DefaultConfig.REDIS_HOST,
                                 port=DefaultConfig.REDIS_PORT,
                                 db=10,
                                 decode_responses=True)
# redis 缓存数据库连接
cache_client = redis.StrictRedis(host=DefaultConfig.REDIS_HOST,
                                 port=DefaultConfig.REDIS_PORT,
                                 db=8,
                                 decode_responses=True)

# # grpc实时排序sparksession
# from pyspark import SparkConf
# from pyspark.sql import SparkSession
# # spark配置
# conf = SparkConf()
# conf.setAll(DefaultConfig.SPARK_GRPC_CONFIG)
#
# SORT_SPARK = SparkSession.builder.config(conf=conf).getOrCreate()