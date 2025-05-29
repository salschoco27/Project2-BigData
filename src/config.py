import os
from dataclasses import dataclass

@dataclass
class KafkaConfig:
    bootstrap_servers = os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092')
    topic = os.getenv('KAFKA_TOPIC', 'retail-transactions')
    retries = int(os.getenv('KAFKA_RETRIES', '5'))
    retry_backoff_ms = int(os.getenv('KAFKA_RETRY_BACKOFF_MS', '1000'))

@dataclass
class SparkConfig:
    app_name = os.getenv('SPARK_APP_NAME', 'RetailConsumer')
    checkpoint_location = os.getenv('CHECKPOINT_LOCATION', '/app/data/output/checkpoint')
    output_path = os.getenv('OUTPUT_PATH', '/app/data/output/retail_batches')
    # Batch berdasarkan WAKTU (setiap 30 detik)
    processing_time = os.getenv('PROCESSING_TIME', '30 seconds')

@dataclass
class ModelConfig:
    models_dir: str = '/app/data/models'
    scaler_path: str = '/app/data/models/scaler.pkl'
    models_info_path: str = '/app/data/models/models_info.json'
    n_clusters: int = 3
    random_state: int = 42

@dataclass
class DataConfig:
    raw_data_path: str = '/app/data/raw/online_retail_II.xlsx'
    batch_data_path: str = '/app/data/output/retail_batches'

# Global config instances
kafka_config = KafkaConfig()
spark_config = SparkConfig()
model_config = ModelConfig()
data_config = DataConfig()