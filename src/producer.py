import time
import random
import json
import logging
from kafka import KafkaProducer
import pandas as pd
from config import kafka_config, data_config

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_producer():
    """Create and return Kafka producer"""
    return KafkaProducer(
        bootstrap_servers=kafka_config.bootstrap_servers,
        value_serializer=lambda x: json.dumps(x).encode('utf-8'),
        retries=kafka_config.retries,
        retry_backoff_ms=kafka_config.retry_backoff_ms
    )

def load_data():
    """Load and clean data"""
    try:
        df = pd.read_excel(data_config.raw_data_path)
        df = df.dropna()
        logger.info(f"Loaded {len(df)} records from {data_config.raw_data_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def send_retail_data():
    """Main function to send retail data to Kafka"""
    producer = create_producer()
    df = load_data()
    
    for index, row in df.iterrows():
        msg = {
            'InvoiceNo': str(row.get('InvoiceNo', '')),
            'StockCode': str(row.get('StockCode', '')),
            'Description': str(row.get('Description', '')),
            'Quantity': int(row.get('Quantity', 0)),
            'InvoiceDate': str(row.get('InvoiceDate', '')),
            'UnitPrice': float(row.get('UnitPrice', 0.0)),
            'CustomerID': str(row.get('Customer ID', '')),
            'Country': str(row.get('Country', ''))
        }
        
        try:
            producer.send(kafka_config.topic, msg)
            logger.info(f"Sent: {msg}")
            time.sleep(random.uniform(0.5, 2.0))
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    producer.close()

if __name__ == "__main__":
    send_retail_data()