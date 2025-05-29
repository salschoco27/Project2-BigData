from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, current_timestamp
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType
import logging
from config import kafka_config, spark_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create Spark session with Kafka packages"""
    return SparkSession.builder \
        .appName(spark_config.app_name) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()

def define_schema():
    """Define the schema for retail data"""
    return StructType() \
        .add("InvoiceNo", StringType()) \
        .add("StockCode", StringType()) \
        .add("Description", StringType()) \
        .add("Quantity", IntegerType()) \
        .add("InvoiceDate", StringType()) \
        .add("UnitPrice", DoubleType()) \
        .add("CustomerID", StringType()) \
        .add("Country", StringType())

def process_kafka_stream():
    """Process Kafka stream and save in batches"""
    spark = create_spark_session()
    schema = define_schema()
    
    # Read from Kafka
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", kafka_config.bootstrap_servers) \
        .option("subscribe", kafka_config.topic) \
        .option("startingOffsets", "earliest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Parse JSON data
    df_parsed = df.selectExpr("CAST(value AS STRING) as json") \
        .select(from_json("json", schema).alias("data")) \
        .select("data.*") \
        .withColumn("processed_time", current_timestamp())

    # Write stream - BATCH BY TIME, bukan by Country
    query = df_parsed.writeStream \
        .format("json") \
        .option("path", spark_config.output_path) \
        .option("checkpointLocation", spark_config.checkpoint_location) \
        .trigger(processingTime=spark_config.processing_time) \
        .outputMode("append") \
        .start()

    query.awaitTermination()

if __name__ == "__main__":
    process_kafka_stream()