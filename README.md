# RLVS Violence Detection — MLOps Pipeline

## Project Overview
Online Continual Learning (OCL) based violence detection system on 
Real-Life Violence Situations (RLVS) dataset, wrapped in a production-grade 
MLOps pipeline.

**Thesis**: Online Continual Learning Adaptation for Violence Detection in Video Streams  
**Student**: Syed Maisum Abbas Rizvi (24i-8007)  
**University**: NUCES-FAST, Islamabad

## Architecture

Kaggle (Training)          EC2 t2.micro (Serving)
─────────────────          ──────────────────────
train.py runs OCL    →     MLflow UI      :5000
MLflow logs locally  →     FastAPI        :8000
Download model.pth   →     Prometheus     :9090
Upload to EC2        →     Grafana        :8080

## Tech Stack
| Component  | Tool              |
|------------|-------------------|
| Training   | PyTorch + Kaggle  |
| Tracking   | MLflow            |
| Serving    | FastAPI           |
| Monitoring | Prometheus        |
| Dashboard  | Grafana           |
| CI/CD      | GitHub Actions    |
| Container  | Docker Compose    |
| Cloud      | AWS EC2 t2.micro  |

## Quick Start

### Local Dev
```bash
git clone https://github.com/YOUR_USERNAME/rlvs-mlops
cd rlvs-mlops
docker-compose up -d
```

### EC2 Setup
```bash
# SSH into EC2
ssh -i your-key.pem ubuntu@YOUR_EC2_IP

# Clone and run
git clone https://github.com/YOUR_USERNAME/rlvs-mlops
cd rlvs-mlops
docker-compose -f docker-compose.prod.yml up -d
```

## MLflow UI
Open: `http://YOUR_EC2_IP:5000`

## Grafana Dashboard
Open: `http://YOUR_EC2_IP:8080`  
Login: admin / admin123

## API Endpoints
- `POST /predict` — Upload video/image for violence detection
- `GET  /health`  — Health check
- `GET  /metrics` — Prometheus metrics
- `GET  /docs`    — Swagger UI

## OCL Variants Evaluated
| Variant        | Avg Acc | Avg Forgetting |
|----------------|---------|----------------|
| ER             | ~86%    | ~12%           |
| ER+MixUp       | ~94%    | ~3.5%          |
| ER+RAR         | ~91%    | ~4.9%          |
| ER+MixUp+RAR   | ~93%    | ~5.3%          |

## CI/CD
Push to `main` → GitHub Actions → Build Docker → Deploy to EC2