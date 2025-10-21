# Docker Compose

[← Quay lại](./README.md) | [← Docker & Dockerfile](./01-docker-dockerfile.md) | [CI/CD →](./03-cicd.md)

## Câu hỏi cơ bản

### 11. Docker Compose là gì và khi nào nên sử dụng?

**Key points:**
- Multi-container applications
- Development và testing environments
- Service orchestration cơ bản
- Declarative configuration
- Khi nào nên dùng Compose vs Kubernetes

### 12. Cấu trúc cơ bản của docker-compose.yml?

**Example:**
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8000:8000"
  database:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: secret
```

**Key components:**
- version
- services
- networks
- volumes
- environment variables

### 13. Giải thích về networking trong Docker Compose?

**Key points:**
- Default bridge network
- Custom networks
- Service discovery by service name
- Port mapping (host:container)
- Network isolation

### 14. Volumes trong Docker Compose hoạt động như thế nào?

**Key points:**
- Named volumes
- Bind mounts
- Data persistence
- Sharing data giữa containers
- Volume drivers

---

## Câu hỏi nâng cao

### 15. Thiết kế docker-compose.yml cho ML pipeline với các components:
- Model training service
- Model serving API
- Database (PostgreSQL)
- Model registry (MLflow)
- Monitoring (Prometheus + Grafana)

**Complete example:**
```yaml
version: '3.8'

services:
  mlflow:
    image: mlflow-server
    ports:
      - "5000:5000"
    environment:
      - BACKEND_STORE_URI=postgresql://user:pass@db:5432/mlflow
      - ARTIFACT_ROOT=/mlflow/artifacts
    volumes:
      - mlflow-data:/mlflow
    depends_on:
      - db

  training:
    build: ./training
    environment:
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    volumes:
      - ./data:/data
      - ./models:/models
    depends_on:
      - mlflow

  api:
    build: ./api
    ports:
      - "8000:8000"
    environment:
      - MODEL_URI=models:/production/latest
      - MLFLOW_TRACKING_URI=http://mlflow:5000
    depends_on:
      - mlflow

  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=mlflow
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
    volumes:
      - db-data:/var/lib/postgresql/data

  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    depends_on:
      - prometheus

volumes:
  mlflow-data:
  db-data:
```

### 16. Environment variables và secrets management trong Docker Compose?

**Methods:**
- .env files
- environment section
- env_file directive
- Docker secrets (swarm mode)
- External secrets management

**Example:**
```yaml
services:
  api:
    environment:
      - DATABASE_URL=${DATABASE_URL}
      - API_KEY=${API_KEY}
    env_file:
      - .env
      - .env.local
```

### 17. Health checks và dependencies trong Docker Compose?

**Key points:**
- healthcheck directive
- depends_on với conditions
- Startup order và wait-for scripts
- Restart policies

**Example:**
```yaml
services:
  db:
    image: postgres:13
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    build: .
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

---

[← Docker & Dockerfile](./01-docker-dockerfile.md) | [CI/CD →](./03-cicd.md)
