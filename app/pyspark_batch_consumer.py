from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StringType, IntegerType, DoubleType

spark = SparkSession.builder.appName("RetailConsumer") \
    .getOrCreate()

schema = StructType() \
    .add("InvoiceNo", StringType()) \
    .add("StockCode", StringType()) \
    .add("Description", StringType()) \
    .add("Quantity", IntegerType()) \
    .add("InvoiceDate", StringType()) \
    .add("UnitPrice", DoubleType()) \
    .add("CustomerID", StringType()) \
    .add("Country", StringType())

df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "kafka:9092") \
    .option("subscribe", "retail-transactions") \
    .option("startingOffsets", "earliest") \
    .load()

df_parsed = df.selectExpr("CAST(value AS STRING) as json") \
    .selectExpr("REPLACE(json, \"'\", '\"') as json") \
    .select(from_json("json", schema).alias("data")) \
    .select("data.*")

query = df_parsed.writeStream \
    .format("json") \
    .option("path", "/app/output/retail_batches") \
    .option("checkpointLocation", "/app/output/checkpoint") \
    .trigger(processingTime="30 seconds") \
    .start()

query.awaitTermination()
