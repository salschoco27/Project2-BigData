from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import json
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Retail Clustering API", version="1.0.0")

# Use hardcoded path that we confirmed works
MODELS_DIR = "/app/data/models"

# Load models with correct path
try:
    model_1 = joblib.load(os.path.join(MODELS_DIR, "model_1.pkl"))
    model_2 = joblib.load(os.path.join(MODELS_DIR, "model_2.pkl"))
    model_3 = joblib.load(os.path.join(MODELS_DIR, "model_3.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    
    # Load model info
    with open(os.path.join(MODELS_DIR, "models_info.json"), "r") as f:
        models_info = json.load(f)
    
    logger.info("All models loaded successfully")
    logger.info(f"Models directory: {MODELS_DIR}")
    logger.info(f"Loaded models: {list(models_info.keys())}")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    logger.error(f"Looking for models in: {MODELS_DIR}")
    import traceback
    traceback.print_exc()
    raise

class RetailItem(BaseModel):
    quantity: float
    price: float

@app.get("/")
def root():
    return {"message": "Retail Clustering API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": True}

@app.get("/models/info")
def get_models_info():
    return models_info

@app.post("/predict/model1")
def predict_cluster_m1(item: RetailItem):
    try:
        # Scale data menggunakan scaler yang sama saat training
        data = np.array([[item.quantity, item.price]])
        data_scaled = scaler.transform(data)
        cluster = model_1.predict(data_scaled)[0]
        
        return {
            "model": "model_1",
            "cluster": int(cluster),
            "input": {"quantity": item.quantity, "price": item.price},
            "silhouette_score": models_info["model_1"]["silhouette_score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/model2")
def predict_cluster_m2(item: RetailItem):
    try:
        data = np.array([[item.quantity, item.price]])
        data_scaled = scaler.transform(data)
        cluster = model_2.predict(data_scaled)[0]
        
        return {
            "model": "model_2",
            "cluster": int(cluster),
            "input": {"quantity": item.quantity, "price": item.price},
            "silhouette_score": models_info["model_2"]["silhouette_score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/model3")
def predict_cluster_m3(item: RetailItem):
    try:
        data = np.array([[item.quantity, item.price]])
        data_scaled = scaler.transform(data)
        cluster = model_3.predict(data_scaled)[0]
        
        return {
            "model": "model_3",
            "cluster": int(cluster),
            "input": {"quantity": item.quantity, "price": item.price},
            "silhouette_score": models_info["model_3"]["silhouette_score"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/best")
def predict_best_model(item: RetailItem):
    try:
        # Pilih model dengan silhouette score tertinggi
        best_model_name = max(models_info.keys(), 
                            key=lambda x: models_info[x]["silhouette_score"])
        
        data = np.array([[item.quantity, item.price]])
        data_scaled = scaler.transform(data)
        
        if best_model_name == "model_1":
            cluster = model_1.predict(data_scaled)[0]
        elif best_model_name == "model_2":
            cluster = model_2.predict(data_scaled)[0]
        else:
            cluster = model_3.predict(data_scaled)[0]
        
        return {
            "model": best_model_name,
            "cluster": int(cluster),
            "input": {"quantity": item.quantity, "price": item.price},
            "silhouette_score": models_info[best_model_name]["silhouette_score"],
            "reason": "best_silhouette_score"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/recommend/products")
def recommend_products_by_cluster(item: RetailItem):
    try:
        # Prediksi dengan model terbaik
        best_model_name = max(models_info.keys(), 
                            key=lambda x: models_info[x]["silhouette_score"])
        
        data = np.array([[item.quantity, item.price]])
        data_scaled = scaler.transform(data)
        
        if best_model_name == "model_1":
            cluster = model_1.predict(data_scaled)[0]
            cluster_centers = model_1.cluster_centers_
        elif best_model_name == "model_2":
            cluster = model_2.predict(data_scaled)[0]
            cluster_centers = model_2.cluster_centers_
        else:
            cluster = model_3.predict(data_scaled)[0]
            cluster_centers = model_3.cluster_centers_
        
        # LOGIC BARU: Interpretasi berdasarkan TOTAL VALUE, bukan asumsi
        total_value = item.quantity * item.price
        
        # Kategorisasi berdasarkan total spending
        if total_value >= 1000:  # $1000+
            customer_type = "VIP Customer"
            recommended_products = [
                "Luxury Jewelry & Watches",
                "High-end Vehicles & Accessories", 
                "Exclusive Art & Collectibles",
                "Premium Real Estate Services",
                "Private Banking Services"
            ]
            marketing_strategy = "White-glove service, exclusive access"
            price_range = "Ultra-luxury ($1000+)"
            
        elif total_value >= 200:  # $200-$999
            customer_type = "Premium Shopper"
            recommended_products = [
                "Designer Clothing & Accessories",
                "High-end Electronics",
                "Luxury Home Decor", 
                "Premium Beauty Products",
                "Gourmet Food & Wine"
            ]
            marketing_strategy = "Focus on quality and exclusivity"
            price_range = "Premium ($200-$999)"
            
        elif total_value >= 50:  # $50-$199
            customer_type = "Regular Shopper"
            recommended_products = [
                "Everyday Clothing & Footwear",
                "Home & Kitchen Essentials",
                "Books & Entertainment",
                "Health & Personal Care",
                "Sporting Goods"
            ]
            marketing_strategy = "Emphasize value and convenience"
            price_range = "Mid-range ($50-$199)"
            
        else:  # Under $50
            customer_type = "Budget Shopper"
            recommended_products = [
                "Basic Clothing & Accessories",
                "Generic/Store Brand Products",
                "Discount Electronics",
                "Budget Home Items",
                "Sale/Clearance Items"
            ]
            marketing_strategy = "Highlight discounts and savings"
            price_range = "Budget ($1-$49)"
        
        # Tambahkan insight berdasarkan quantity vs price pattern
        if item.quantity > 20 and item.price < 10:
            purchase_pattern = "Bulk Buying Pattern"
        elif item.quantity < 5 and item.price > 50:
            purchase_pattern = "Premium Quality Pattern"
        elif item.quantity > 10 and item.price > 50:
            purchase_pattern = "High-Value Bulk Pattern"
        else:
            purchase_pattern = "Standard Purchase Pattern"
        
        return {
            "customer_cluster": int(cluster),
            "customer_type": customer_type,
            "purchase_pattern": purchase_pattern,
            "current_purchase": {
                "quantity": item.quantity,
                "price": item.price,
                "total_value": round(total_value, 2)
            },
            "recommended_products": recommended_products,
            "marketing_strategy": marketing_strategy,
            "recommended_price_range": price_range,
            "model_used": best_model_name,
            "confidence": models_info[best_model_name]["silhouette_score"],
            "cluster_center": cluster_centers[cluster].tolist()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))