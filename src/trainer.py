import os
import logging
from datetime import datetime
import json
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import joblib
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_spark_session():
    """Create Spark session"""
    return SparkSession.builder \
        .appName("ModelTrainer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .getOrCreate()

def load_batch_data():
    """Load all batch data using Spark"""
    try:
        spark = create_spark_session()
        batch_path = "/app/data/output/retail_batches"
        
        if not os.path.exists(batch_path):
            logger.error("Batch data directory not found")
            return pd.DataFrame()
        
        # Check if there are any JSON files
        files = []
        for root, dirs, filenames in os.walk(batch_path):
            for filename in filenames:
                if filename.endswith('.json'):
                    files.append(os.path.join(root, filename))
        
        if not files:
            logger.warning("No JSON files found in batch directory")
            return pd.DataFrame()
        
        # Read all JSON files with Spark
        df_spark = spark.read.option("multiline", "true").json(batch_path)
        
        # Convert to Pandas for sklearn
        df_pandas = df_spark.toPandas()
        
        logger.info(f"Loaded {len(df_pandas)} records from Spark")
        spark.stop()
        
        return df_pandas
        
    except Exception as e:
        logger.error(f"Error loading data with Spark: {e}")
        logger.info("Falling back to pandas...")
        
        # Fallback to pandas if Spark fails
        try:
            files = []
            for root, dirs, filenames in os.walk("/app/data/output/retail_batches"):
                for filename in filenames:
                    if filename.endswith('.json'):
                        files.append(os.path.join(root, filename))
            
            if not files:
                return pd.DataFrame()
            
            all_data = pd.concat([pd.read_json(f, lines=True) for f in files], ignore_index=True)
            logger.info(f"Loaded {len(all_data)} records with pandas fallback")
            return all_data
        except Exception as e2:
            logger.error(f"Pandas fallback also failed: {e2}")
            return pd.DataFrame()

def prepare_features(data):
    """Prepare features for clustering"""
    # Use UnitPrice instead of UnitPrice if column name differs
    price_col = 'UnitPrice' if 'UnitPrice' in data.columns else 'price'
    quantity_col = 'Quantity' if 'Quantity' in data.columns else 'quantity'
    
    X = data[[quantity_col, price_col]].fillna(0)
    
    # Convert to numeric if needed
    X[quantity_col] = pd.to_numeric(X[quantity_col], errors='coerce').fillna(0)
    X[price_col] = pd.to_numeric(X[price_col], errors='coerce').fillna(0)
    
    # Remove outliers
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    X = X[~((X < (Q1 - 1.5 * IQR)) | (X > (Q3 + 1.5 * IQR))).any(axis=1)]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

def train_models():
    """Train multiple clustering models"""
    all_data = load_batch_data()
    
    if all_data.empty:
        logger.error("No data available for training")
        return
    
    logger.info(f"Data columns: {list(all_data.columns)}")
    logger.info(f"Data shape: {all_data.shape}")
    
    X_scaled, scaler = prepare_features(all_data)
    
    # Create models directory
    os.makedirs("/app/data/models", exist_ok=True)
    
    # Save scaler
    joblib.dump(scaler, "/app/data/models/scaler.pkl")
    
    # Train 3 models with different data portions
    split_size = len(X_scaled) // 3
    models_info = {}
    
    for i in range(1, 4):
        if i == 1:
            data_slice = X_scaled[:split_size]
        elif i == 2:
            data_slice = X_scaled[:2*split_size]
        else:
            data_slice = X_scaled
        
        if len(data_slice) < 3:
            logger.warning(f"Not enough data for model {i}, skipping...")
            continue
        
        model = KMeans(n_clusters=min(3, len(data_slice)), random_state=42, n_init=10)
        model.fit(data_slice)
        
        score = silhouette_score(data_slice, model.labels_)
        joblib.dump(model, f"/app/data/models/model_{i}.pkl")
        
        models_info[f'model_{i}'] = {
            'data_size': len(data_slice),
            'silhouette_score': float(score),
            'training_time': datetime.now().isoformat()
        }
        
        logger.info(f"Model {i}: {len(data_slice)} samples, silhouette_score: {score:.3f}")
    
    # Save model metadata
    with open("/app/data/models/models_info.json", 'w') as f:
        json.dump(models_info, f, indent=2)
    
    logger.info("All models trained successfully")

if __name__ == "__main__":
    train_models()