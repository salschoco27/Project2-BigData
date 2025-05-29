# Project2-BigData

Real-time retail data streaming and machine learning pipeline menggunakan Kafka, Spark, dan FastAPI untuk clustering analysis.

## Anggota Kelompok

- Salsabila Rahmah - 5027231005
- Chelsea Vania Haryono - 5027231003
- Adlya Isriena Aftarisya - 5027231066

### Components:

1. **Kafka Producer** ([src/producer.py](src/producer.py)): Streaming data retail dari Excel file ke Kafka dengan random delay
2. **Spark Consumer** ([src/consumer.py](src/consumer.py)): Membaca stream dari Kafka dan menyimpan dalam batch
3. **Model Trainer** ([src/trainer.py](src/trainer.py)): Training 3 model clustering dengan data portions berbeda
4. **FastAPI** ([src/api.py](src/api.py)): REST API untuk inference menggunakan trained models

## 📊 Data Pipeline

### Input Data
- **Source**: [data/raw/online_retail_II.xlsx](data/raw/online_retail_II.xlsx)
- **Format**: Excel file dengan data transaksi retail
- **Fields**: InvoiceNo, StockCode, Description, Quantity, InvoiceDate, UnitPrice, CustomerID, Country

### Data Flow
1. **Producer** streams data row-by-row dengan random delay (0.5-2.0 detik)
2. **Consumer** membaca stream dan menyimpan dalam batches setiap 30 detik
3. **Trainer** menggunakan batch data untuk training 3 model:
   - **Model 1**: 1/3 data pertama
   - **Model 2**: 2/3 data pertama
   - **Model 3**: Semua data

### Model Training Strategy
```
Data: [=====================================] (100%)
Model 1: [============]                        (33%)
Model 2: [========================]            (67%)
Model 3: [====================================] (100%)
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.10+ (jika running locally)
- Ports available: 8000, 9092, 2181, 4040

### Option 1: Docker Compose (Recommended)

```bash
# Clone repository
git clone <your-repo-url>
cd Project2-BigData

# Start complete pipeline
docker-compose up -d

# Check services status
docker-compose ps
```

### Option 2: Manual Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Start Kafka infrastructure
docker-compose up -d zookeeper kafka

# Wait for Kafka to be ready (30 seconds)
sleep 30

# Run components in separate terminals:

# Terminal 1: Producer
python src/producer.py

# Terminal 2: Consumer
python src/consumer.py

# Terminal 3: Trainer (after some batches are created)
python src/trainer.py

# Terminal 4: API
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

## 📡 API Endpoints

Base URL: `http://localhost:8000`

### Health & Info
- `GET /` - API information dan status
- `GET /health` - Health check
- `GET /models/info` - Model metadata dan performance metrics

### Clustering Predictions
- `POST /predict/model1` - Prediksi menggunakan Model 1
- `POST /predict/model2` - Prediksi menggunakan Model 2  
- `POST /predict/model3` - Prediksi menggunakan Model 3
- `POST /predict/best` - Prediksi menggunakan model dengan performa terbaik

### Request Format
```json
{
  "quantity": 5.0,
  "price": 10.5
}
```

### Response Format
```json
{
  "model": "model_1",
  "cluster": 2,
  "input": {
    "quantity": 5.0,
    "price": 10.5
  },
  "silhouette_score": 0.742
}
```

## 🧪 Testing API

### Using cURL
```bash
# Test health endpoint
curl -X GET http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict/model1 \
  -H "Content-Type: application/json" \
  -d '{"quantity": 5, "price": 10.5}'

# Test best model
curl -X POST http://localhost:8000/predict/best \
  -H "Content-Type: application/json" \
  -d '{"quantity": 3, "price": 25.0}'
```

### Using Python
```python
import requests

# API endpoint
url = "http://localhost:8000/predict/best"

# Test data
data = {"quantity": 5, "price": 10.5}

# Make request
response = requests.post(url, json=data)
print(response.json())
```

### Interactive API Documentation
Visit: `http://localhost:8000/docs`

## 📁 Project Structure

```
Project2-BigData/
├── docker-compose.yml          # Docker orchestration
├── requirements.txt            # Python dependencies
├── README.md                  # This file
├── .env                       # Environment variables
├── .gitignore                 # Git ignore rules
├── data/
│   ├── raw/                   # Input data
│   │   └── online_retail_II.xlsx
│   ├── output/                # Processed data
│   │   ├── retail_batches/    # Batch files from Spark
│   │   └── checkpoint/        # Spark checkpoints
│   └── models/                # Trained models
│       ├── model_1.pkl
│       ├── model_2.pkl
│       ├── model_3.pkl
│       ├── scaler.pkl
│       └── models_info.json
├── src/                       # Source code
│   ├── producer.py           # Kafka producer
│   ├── consumer.py           # Spark consumer
│   ├── trainer.py            # Model trainer
│   ├── api.py                # FastAPI application
│   └── config.py             # Configuration
└── docker/                    # Docker files
    ├── Dockerfile.producer
    ├── Dockerfile.consumer
    ├── Dockerfile.trainer
    └── Dockerfile.api
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC=retail-transactions

# Spark Configuration
SPARK_MASTER=local[*]
SPARK_APP_NAME=RetailConsumer

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Data Paths
DATA_RAW_PATH=data/raw/online_retail_II.xlsx
DATA_OUTPUT_PATH=data/output/retail_batches
MODELS_PATH=data/models
```

### Docker Services
- **zookeeper**: Port 2181
- **kafka**: Port 9092  
- **producer**: Streams data to Kafka
- **consumer**: Processes streams with Spark
- **trainer**: Trains ML models
- **api**: REST API on port 8000

## 📊 Monitoring

### Check Service Status
```bash
# Docker containers
docker-compose ps

# Service logs
docker-compose logs -f producer
docker-compose logs -f consumer
docker-compose logs -f api

# Kafka topics
docker exec -it <kafka-container> kafka-topics --list --bootstrap-server localhost:9092
```

### Data Monitoring
```bash
# Check generated batches
ls -la data/output/retail_batches/

# Check trained models
ls -la data/models/

# Model performance
curl http://localhost:8000/models/info
```

## 🐛 Troubleshooting

### Common Issues

1. **Kafka Connection Error**
   ```bash
   # Wait for Kafka to be ready
   docker-compose logs kafka
   # Restart if needed
   docker-compose restart kafka
   ```

2. **Model Not Found Error**
   ```bash
   # Train models first
   docker-compose up trainer
   # Or run manually
   python src/trainer.py
   ```

3. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   # Kill the process or change port
   ```

4. **Spark Connection Issues**
   ```bash
   # Check Spark UI (if available)
   # http://localhost:4040
   # Check consumer logs
   docker-compose logs consumer
   ```

### Debug Mode
```bash
# Run with verbose logging
export PYTHONPATH=.
python -m src.api --log-level debug

# Check specific component
docker-compose up --no-deps producer
```

## 🤝 Development

### Adding New Features

1. **New API Endpoint**:
   - Add function to [src/api.py](src/api.py)
   - Update documentation

2. **New Model**:
   - Modify [src/trainer.py](src/trainer.py)
   - Update API loading logic

3. **New Data Source**:
   - Modify [src/producer.py](src/producer.py)
   - Update schema in [src/consumer.py](src/consumer.py)

### Testing
```bash
# Install dev dependencies
pip install pytest httpx

# Run tests (if implemented)
pytest tests/

# Manual API testing
python tests/test_api_manual.py
```

## 📝 Model Performance

Current models menggunakan K-Means clustering dengan:
- **Features**: Quantity dan UnitPrice (scaled)
- **Clusters**: 3 clusters per model
- **Metric**: Silhouette Score
- **Best Model**: Model dengan highest silhouette score

### Example Performance
```json
{
  "model_1": {
    "data_size": 178234,
    "silhouette_score": 0.742,
    "training_time": "2024-01-15T10:30:00"
  },
  "model_2": {
    "data_size": 356468,
    "silhouette_score": 0.758,
    "training_time": "2024-01-15T10:32:00"
  },
  "model_3": {
    "data_size": 534702,
    "silhouette_score": 0.771,
    "training_time": "2024-01-15T10:35:00"
  }
}
```