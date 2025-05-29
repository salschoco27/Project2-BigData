import time
import random
import csv
from kafka import KafkaProducer
import pandas as pd

producer = KafkaProducer(bootstrap_servers='kafka:9092')
topic = 'retail-transactions'

df = pd.read_excel('online_retail_II.xlsx')
df.to_csv('retail_dataset.csv', index=False)

with open('retail_dataset.csv') as file:
    reader = csv.DictReader(file)
    for row in reader:
        msg = str(row).encode('utf-8')
        producer.send(topic, msg)
        print(f"Sent: {msg}")
        time.sleep(random.uniform(0.5, 2.0))  # Simulate streaming
