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
    
@app.post("/analyze/customer-segment")
def analyze_customer_segment(item: RetailItem):
    """
    Analisis segmentasi customer berdasarkan purchase behavior
    """
    try:
        # Prediksi cluster dengan model terbaik
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
        
        total_value = item.quantity * item.price
        
        # Analisis behavior patterns
        price_per_item = item.price
        if price_per_item > 100:
            price_category = "Premium"
            price_sensitivity = "Low"
        elif price_per_item > 25:
            price_category = "Mid-range"
            price_sensitivity = "Medium"
        else:
            price_category = "Budget"
            price_sensitivity = "High"
        
        if item.quantity > 20:
            quantity_category = "High Volume"
            purchase_frequency = "Bulk Buyer"
        elif item.quantity > 5:
            quantity_category = "Medium Volume"
            purchase_frequency = "Regular Buyer"
        else:
            quantity_category = "Low Volume"
            purchase_frequency = "Selective Buyer"
        
        # Customer lifecycle stage
        if total_value > 1000:
            lifecycle_stage = "VIP/Loyalty"
            retention_priority = "High"
        elif total_value > 200:
            lifecycle_stage = "Growth"
            retention_priority = "Medium"
        elif total_value > 50:
            lifecycle_stage = "Development"
            retention_priority = "Medium"
        else:
            lifecycle_stage = "Acquisition"
            retention_priority = "Low"
        
        # Marketing persona
        if price_category == "Premium" and quantity_category == "Low Volume":
            marketing_persona = "Quality Seeker"
            preferred_channels = ["Email", "Direct Mail", "Personal Consultation"]
        elif price_category == "Budget" and quantity_category == "High Volume":
            marketing_persona = "Value Hunter"
            preferred_channels = ["SMS", "App Notifications", "Social Media"]
        elif total_value > 500:
            marketing_persona = "High Spender"
            preferred_channels = ["VIP Events", "Personal Account Manager"]
        else:
            marketing_persona = "Average Consumer"
            preferred_channels = ["Email", "Website", "Social Media"]
        
        return {
            "customer_cluster": int(cluster),
            "segment_analysis": {
                "price_category": price_category,
                "quantity_category": quantity_category,
                "price_sensitivity": price_sensitivity,
                "purchase_frequency": purchase_frequency,
                "lifecycle_stage": lifecycle_stage,
                "marketing_persona": marketing_persona
            },
            "business_metrics": {
                "total_value": round(total_value, 2),
                "retention_priority": retention_priority,
                "upsell_potential": "High" if total_value > 200 else "Medium" if total_value > 50 else "Low",
                "cross_sell_opportunity": "Strong" if cluster in [0, 1] else "Moderate"
            },
            "marketing_strategy": {
                "preferred_channels": preferred_channels,
                "message_tone": "Exclusive" if price_category == "Premium" else "Value-focused",
                "promotion_type": "Quality-based" if price_sensitivity == "Low" else "Discount-based"
            },
            "model_used": best_model_name,
            "confidence": models_info[best_model_name]["silhouette_score"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/customer-lifetime-value")
def predict_customer_lifetime_value(item: RetailItem):
    """
    Prediksi Customer Lifetime Value dan retention metrics
    """
    try:
        # Prediksi cluster
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
        
        current_value = item.quantity * item.price
        
        # CLV calculation berdasarkan cluster dan purchase pattern
        if cluster == 0:  # Typically premium customers
            clv_multiplier = 4.5
            retention_rate = 0.85
            purchase_frequency_per_year = 12
            average_order_value_growth = 1.15
            churn_risk = "Low"
        elif cluster == 1:  # Regular customers
            clv_multiplier = 2.8
            retention_rate = 0.70
            purchase_frequency_per_year = 8
            average_order_value_growth = 1.05
            churn_risk = "Medium"
        else:  # Bulk/price-sensitive customers
            clv_multiplier = 3.2
            retention_rate = 0.60
            purchase_frequency_per_year = 6
            average_order_value_growth = 1.02
            churn_risk = "High"
        
        # Adjust based on current purchase value
        if current_value > 1000:
            clv_multiplier *= 1.5
            retention_rate = min(0.95, retention_rate + 0.1)
            churn_risk = "Very Low"
        elif current_value > 500:
            clv_multiplier *= 1.3
            retention_rate = min(0.90, retention_rate + 0.05)
            if churn_risk == "High":
                churn_risk = "Medium"
        elif current_value < 25:
            clv_multiplier *= 0.7
            retention_rate = max(0.4, retention_rate - 0.1)
            churn_risk = "High"
        
        # Calculate metrics
        estimated_clv = current_value * clv_multiplier * retention_rate
        annual_value = current_value * purchase_frequency_per_year
        three_year_value = annual_value * 3 * (retention_rate ** 2)
        
        # Customer value tier
        if estimated_clv > 2000:
            value_tier = "Platinum"
            service_level = "White Glove"
        elif estimated_clv > 1000:
            value_tier = "Gold"
            service_level = "Premium"
        elif estimated_clv > 500:
            value_tier = "Silver"
            service_level = "Enhanced"
        else:
            value_tier = "Bronze"
            service_level = "Standard"
        
        # Retention strategies
        if churn_risk == "High":
            retention_actions = [
                "Immediate discount offer",
                "Personal outreach",
                "Loyalty program enrollment",
                "Product usage optimization"
            ]
        elif churn_risk == "Medium":
            retention_actions = [
                "Targeted email campaign",
                "Cross-sell recommendations",
                "Feedback survey",
                "Special offers"
            ]
        else:
            retention_actions = [
                "VIP recognition program",
                "Early access to new products",
                "Referral rewards",
                "Exclusive events"
            ]
        
        return {
            "customer_cluster": int(cluster),
            "current_purchase_value": round(current_value, 2),
            "lifetime_value_prediction": {
                "estimated_clv": round(estimated_clv, 2),
                "annual_value_estimate": round(annual_value, 2),
                "three_year_value": round(three_year_value, 2),
                "value_tier": value_tier
            },
            "retention_metrics": {
                "retention_probability": retention_rate,
                "churn_risk": churn_risk,
                "purchase_frequency_per_year": purchase_frequency_per_year,
                "aov_growth_rate": average_order_value_growth
            },
            "business_recommendations": {
                "service_level": service_level,
                "retention_actions": retention_actions,
                "investment_priority": "High" if estimated_clv > 1000 else "Medium" if estimated_clv > 500 else "Low",
                "marketing_budget_allocation": f"{min(15, max(2, int(estimated_clv/100)))}% of CLV"
            },
            "model_used": best_model_name,
            "confidence": models_info[best_model_name]["silhouette_score"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))