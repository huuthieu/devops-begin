# Docker & Dockerfile

[← Quay lại](./README.md)

## Câu hỏi cơ bản

### 1. Docker là gì và tại sao nó quan trọng trong MLOps?

**Câu trả lời:**

Docker là một nền tảng containerization cho phép đóng gói ứng dụng cùng với tất cả các dependencies (libraries, runtime, code) vào một container độc lập. Container này có thể chạy nhất quán trên bất kỳ môi trường nào.

**Tại sao quan trọng trong MLOps:**

1. **Reproducibility (Tái tạo lại kết quả):**
   - Đảm bảo model training/inference hoạt động giống nhất trên dev, staging, production
   - Mọi người dùng cùng một môi trường, dependency versions, OS
   - Tránh được "works on my machine" problem

2. **Isolation (Cô lập):**
   - Các ứng dụng khác nhau không can thiệp đến nhau
   - Mỗi model có thể có version dependencies riêng
   - Resource management rõ ràng (CPU, memory limits)

3. **Portability (Tính di động):**
   - Chạy trên Linux, Windows, Mac như nhau
   - Deploy từ laptop lên cloud (AWS, GCP, Azure) không thay đổi code
   - Dễ scale horizontal với orchestration tools (Kubernetes)

4. **ML Pipeline Efficiency:**
   - Model training, validation, serving trong containers riêng biệt
   - CI/CD integration: tự động test, build, deploy models
   - Version control cho environments (Dockerfile như source code)
   - Dễ setup complex pipelines với Docker Compose

**Ví dụ real-world:**
```
Data Preparation Container → Model Training Container →
Model Serving Container → Monitoring Container
```

**Key points:**
- Giải thích về containerization
- Lợi ích: reproducibility, isolation, portability
- Vai trò trong ML workflow

### 2. Sự khác biệt giữa Docker Image và Docker Container?

**Câu trả lời:**

| Tiêu chí | Docker Image | Docker Container |
|---------|------------|-----------------|
| **Định nghĩa** | Template/Blueprint chứa tất cả code, dependencies | Running instance của một image |
| **Tính chất** | Immutable (không thay đổi) | Mutable, có writable layer riêng |
| **Lưu trữ** | File trên disk | Process running trong OS |
| **Kích thước** | Nhẹ (hàng MB đến GB) | Nặng hơn do writable layer |
| **Lifecycle** | Tồn tại cho đến khi xóa | Tạo từ image, có thể stop/start |
| **Multiple instances** | 1 image có thể tạo nhiều containers | Mỗi container là instance độc lập |
| **Reusability** | Có thể share giữa team | Ephemeral, thường xóa sau dùng |

**Ví dụ minh họa:**

```
Docker Image: python:3.9-slim
    ↓ (docker run)
Container 1: model-training-job (đang chạy)
Container 2: model-serving-app (đang chạy)
Container 3: data-prep-job (đã dừng)
```

**Lifecycle chi tiết:**

```
# Tạo image từ Dockerfile
$ docker build -t my-model:v1 .

# Tạo container từ image (multiple instances)
$ docker run -d my-model:v1              # Container 1
$ docker run -d my-model:v1              # Container 2

# Container lifecycle
Running → Paused → Stopped → Removed

# Xem log, exec, inspect từng container riêng
$ docker logs container-id
$ docker exec -it container-id bash
```

**Lợi ích của design này trong MLOps:**
- Dễ scale: 1 image → nhiều containers chạy model serving
- Experiment: Tạo containers tạm thời cho testing
- Version control: Quản lý image versions như git
- Resource isolation: Mỗi container có resource limits riêng

**Key points:**
- Image: template, immutable
- Container: running instance của image
- Lifecycle và quản lý

### 3. Giải thích cấu trúc của một Dockerfile?

**Câu trả lời:**

Dockerfile là một text file chứa các instructions để build Docker image. Mỗi instruction tạo ra một layer trong image, xếp chồng lên nhau.

**Các lệnh chính:**

```dockerfile
# 1. BASE IMAGE: Xuất phát từ image nào
FROM python:3.9-slim

# 2. METADATA: Thông tin về image
LABEL maintainer="mlops-team@company.com"
LABEL version="1.0"

# 3. ENVIRONMENT VARIABLES: Set biến môi trường
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models

# 4. WORKING DIRECTORY: Thư mục làm việc
WORKDIR /app

# 5. COPY/ADD: Copy files từ host vào container
COPY requirements.txt .
COPY src/ ./src/
COPY model/ ./model/

# 6. RUN: Chạy commands trong build time
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y curl

# 7. EXPOSE: Khai báo ports (documentation)
EXPOSE 8000

# 8. HEALTHCHECK: Kiểm tra sức khỏe container
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# 9. ENTRYPOINT: Main command (không override được dễ)
ENTRYPOINT ["python"]

# 10. CMD: Default arguments cho ENTRYPOINT (có thể override)
CMD ["src/app.py"]
```

**Giải thích chi tiết:**

| Instruction | Mục đích | Ví dụ |
|-----------|---------|-------|
| **FROM** | Base image để bắt đầu | `FROM ubuntu:20.04` |
| **RUN** | Chạy command (build time) | `RUN pip install numpy` |
| **COPY** | Copy files từ host | `COPY . /app` |
| **ADD** | Like COPY nhưng có thêm tính năng | `ADD archive.tar.gz /app` |
| **WORKDIR** | Set working directory | `WORKDIR /app` |
| **ENV** | Set environment variables | `ENV DEBUG=true` |
| **EXPOSE** | Expose ports (documentation) | `EXPOSE 8080` |
| **CMD** | Default command (overridable) | `CMD ["python", "app.py"]` |
| **ENTRYPOINT** | Main executable | `ENTRYPOINT ["python"]` |
| **USER** | Run as user | `USER appuser` |
| **VOLUME** | Mount point | `VOLUME ["/data"]` |
| **HEALTHCHECK** | Container health check | `HEALTHCHECK CMD curl ...` |

**Ví dụ ML Model Dockerfile:**

```dockerfile
FROM python:3.9-slim

# Metadata
LABEL author="ML Team"

# Set environment
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/models/model.pkl
ENV PORT=8000

# Set working directory
WORKDIR /app

# Copy dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY models/ ./models/

# Non-root user
RUN useradd -m -u 1000 mluser
USER mluser

# Health check
HEALTHCHECK --interval=30s CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Default command
CMD ["python", "src/api.py"]
```

**Thứ tự instructions quan trọng:**
- Base image → Environment → Dependencies → Application code
- Maximize layer caching: ít thay đổi trước, thường xuyên thay đổi sau

**Key points:**
- Base image (FROM)
- Working directory (WORKDIR)
- Copy files (COPY, ADD)
- Install dependencies (RUN)
- Environment variables (ENV)
- Entrypoint và CMD

### 4. Phân biệt CMD và ENTRYPOINT trong Dockerfile?

**Câu trả lời:**

CMD và ENTRYPOINT đều dùng để chỉ định default command, nhưng cách xử lý khác nhau.

**So sánh chi tiết:**

| Tiêu chí | CMD | ENTRYPOINT |
|---------|-----|-----------|
| **Override** | Dễ override khi `docker run` | Khó override, append arguments |
| **Cách sử dụng** | Chỉ có 1 CMD trong Dockerfile | Chỉ có 1 ENTRYPOINT trong Dockerfile |
| **Kiểu** | Shell form hoặc Exec form | Shell form hoặc Exec form |
| **Khi nào chọn** | Default behavior | Fixed behavior |
| **Ứng dụng** | Default parameters | Wrapping scripts |

**Ví dụ 1: Chỉ dùng CMD**

```dockerfile
# Dockerfile 1: Chỉ CMD
FROM python:3.9
CMD ["python", "app.py"]
```

```bash
# Thực thi mặc định
$ docker run myimage
# Kết quả: python app.py

# Override CMD
$ docker run myimage python other_script.py
# Kết quả: python other_script.py
```

**Ví dụ 2: Chỉ dùng ENTRYPOINT**

```dockerfile
# Dockerfile 2: Chỉ ENTRYPOINT
FROM python:3.9
ENTRYPOINT ["python"]
```

```bash
# Thực thi mặc định (cần args)
$ docker run myimage
# Kết quả: python (lỗi vì thiếu script)

# Thêm arguments
$ docker run myimage app.py
# Kết quả: python app.py

# Override ENTRYPOINT (cần --entrypoint)
$ docker run --entrypoint java myimage MyApp
# Kết quả: java MyApp
```

**Ví dụ 3: Kết hợp cả hai (BEST PRACTICE)**

```dockerfile
# Dockerfile 3: ENTRYPOINT + CMD
FROM python:3.9
ENTRYPOINT ["python"]
CMD ["app.py"]
```

```bash
# Mặc định: python app.py
$ docker run myimage

# Override CMD (append to ENTRYPOINT)
$ docker run myimage src/train.py
# Kết quả: python src/train.py

# Override ENTRYPOINT
$ docker run --entrypoint java myimage MyApp.jar
# Kết quả: java MyApp.jar
```

**Ví dụ 4: ML Model Training Job**

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/
COPY data/ ./data/

# ENTRYPOINT: Main executable
ENTRYPOINT ["python", "src/train.py"]

# CMD: Default arguments
CMD ["--model", "random_forest", "--epochs", "100"]
```

```bash
# Mặc định: python src/train.py --model random_forest --epochs 100
$ docker run training-image

# Override model chỉ
$ docker run training-image --model xgboost
# Kết quả: python src/train.py --model xgboost

# Override hoàn toàn (dùng --entrypoint)
$ docker run --entrypoint python training-image -c "print('test')"
```

**Best Practices:**

1. **Dùng Exec form** (mảng JSON) thay Shell form:
   ```dockerfile
   # Tốt: Exec form
   ENTRYPOINT ["python", "app.py"]

   # Kém: Shell form (không nhận SIGTERM)
   ENTRYPOINT python app.py
   ```

2. **Kết hợp ENTRYPOINT + CMD khi:**
   - Bạn muốn fixed entry point (e.g., always run python)
   - Nhưng cho phép override parameters dễ dàng

3. **Wrapper script pattern:**
   ```dockerfile
   FROM python:3.9
   COPY entrypoint.sh /
   ENTRYPOINT ["/entrypoint.sh"]
   CMD ["python", "app.py"]
   ```

**Key points:**
- CMD: default command, có thể override
- ENTRYPOINT: main command, append arguments
- Best practices khi kết hợp cả hai

### 5. Multi-stage builds trong Docker là gì? Tại sao sử dụng?

**Câu trả lời:**

Multi-stage build cho phép sử dụng nhiều FROM statements trong một Dockerfile. Mỗi stage có thể copy artifacts từ stage trước đó, nhưng chỉ final stage được include trong final image.

**Lợi ích:**

1. **Giảm kích thước image:** Loại bỏ build dependencies khỏi final image
2. **Tách concerns:** Build stage vs Runtime stage
3. **Security:** Không cần build tools trong production
4. **Performance:** Final image nhẹ hơn → deploy nhanh hơn

**Ví dụ: Build ML Model**

```dockerfile
# Stage 1: Model Training/Compilation
FROM pytorch:2.0-cuda11.8 AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY data/ ./data/

# Compile model (heavy operation)
RUN python src/train.py --save-path=/build/model.pkl
RUN python src/optimize.py --input=/build/model.pkl --output=/build/model.onnx

# Stage 2: Runtime (final image)
FROM python:3.9-slim

WORKDIR /app

# Chỉ copy model từ builder, không cần source code
COPY --from=builder /build/model.pkl ./models/
COPY --from=builder /build/model.onnx ./models/

# Runtime dependencies chỉ
COPY requirements-prod.txt .
RUN pip install -r requirements-prod.txt

COPY src/api.py ./

EXPOSE 8000
CMD ["python", "api.py"]
```

**So sánh kích thước:**

```
Single-stage Dockerfile:
- PyTorch base image: ~3GB
- Training dependencies: +500MB
- Source code: +100MB
- Model files: +200MB
- Total: ~3.8GB ❌

Multi-stage Dockerfile:
- Builder stage: 3.8GB (không trong final image)
- Python slim base: 150MB
- Model files: 200MB
- Runtime deps: 50MB
- Total: ~400MB ✅ (10x nhỏ hơn!)
```

**Ví dụ 2: Golang Model Server (compiled language)**

```dockerfile
# Stage 1: Build
FROM golang:1.20 AS builder
WORKDIR /build
COPY . .
RUN go build -o server main.go

# Stage 2: Runtime
FROM scratch
COPY --from=builder /build/server /
EXPOSE 8080
ENTRYPOINT ["/server"]
```

**Ví dụ 3: ML Pipeline (3 stages)**

```dockerfile
# Stage 1: Data Preparation
FROM python:3.9 AS data_prep
WORKDIR /data
COPY scripts/prepare.py .
COPY raw_data/ ./raw_data/
RUN python prepare.py

# Stage 2: Model Training
FROM pytorch:2.0 AS training
WORKDIR /train
COPY --from=data_prep /data/processed_data ./data/
COPY src/train.py .
RUN python train.py --output=/train/model.pkl

# Stage 3: Model Serving
FROM python:3.9-slim
WORKDIR /app
COPY --from=training /train/model.pkl ./models/
COPY src/inference.py .
COPY requirements.txt .
RUN pip install -r requirements.txt
CMD ["python", "inference.py"]
```

**Advanced: Conditional stages**

```dockerfile
# Build arguments
ARG BUILD_ENV=production

# Stage 1: Development
FROM python:3.9 AS dev
WORKDIR /app
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt
COPY . .

# Stage 2: Production
FROM python:3.9-slim AS prod
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/

# Final stage: Select based on argument
FROM ${BUILD_ENV} as final
```

```bash
# Build for production
$ docker build --target prod -t myapp:prod .

# Build for development
$ docker build --target dev -t myapp:dev .
```

**Real-world MLOps Example:**

```dockerfile
# Stage 1: Feature Engineering
FROM python:3.9 AS features
RUN pip install pandas numpy scikit-learn
COPY src/features.py .
COPY raw_data/ ./data/
RUN python features.py

# Stage 2: Model Training
FROM python:3.9 AS training
RUN pip install torch xgboost hyperopt
COPY --from=features /features.pkl .
COPY src/train.py .
RUN python train.py

# Stage 3: Production Serving
FROM python:3.9-slim
RUN pip install fastapi uvicorn numpy
COPY --from=training /model.pkl ./models/
COPY src/api.py .
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0"]
```

**Key points:**
- Giảm kích thước image
- Tách build và runtime dependencies
- Ví dụ: compile model trong stage 1, deploy trong stage 2

---

## Câu hỏi nâng cao

### 6. Làm thế nào để optimize Docker image cho ML models?

**Câu trả lời:**

Optimization Docker image rất quan trọng trong MLOps vì nhỏ hơn → deploy nhanh hơn → tiết kiệm storage/bandwidth.

**1. Chọn Base Image Nhỏ**

```dockerfile
# ❌ Quá lớn: 1.2GB
FROM ubuntu:latest
FROM python:3.9
FROM pytorch/pytorch:latest

# ✅ Tối ưu hơn: 150MB
FROM python:3.9-slim
FROM python:3.9-alpine

# ✅ Siêu nhỏ cho compiled binaries: 5MB
FROM scratch
```

**Bảng so sánh base images:**

| Base Image | Kích thước | Ưu điểm | Nhược điểm |
|-----------|-----------|--------|-----------|
| `ubuntu:latest` | 77MB | Quen thuộc, tools đầy đủ | Lớn nhất |
| `python:3.9` | 886MB | Full Python | Nhiều dependencies |
| `python:3.9-slim` | 151MB | Python + essentials | Thiếu build tools |
| `python:3.9-alpine` | 50MB | Siêu nhỏ | Thiếu nhiều tools |
| `scratch` | 0MB | Tuyệt vời cho binaries | Chỉ cho compiled code |

**2. Multi-stage Build**

```dockerfile
# Giảm từ 1.5GB → 200MB
FROM pytorch:2.0 AS builder
RUN pip install torch numpy scikit-learn
COPY train.py .
RUN python train.py

FROM python:3.9-slim
COPY --from=builder /model.pkl ./
CMD ["python", "serve.py"]
```

**3. .dockerignore File**

```
# .dockerignore
.git
.gitignore
.DS_Store
__pycache__
*.pyc
*.pyo
*.egg-info
.pytest_cache
.venv
venv/
notebooks/
*.ipynb
raw_data/
*.zip
*.tar.gz
.env
.env.local
node_modules/
```

**4. Layer Caching Strategy**

```dockerfile
FROM python:3.9-slim

# ❌ BAD: Thay requirements → bust cache
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt

# ✅ GOOD: Ít thay đổi trước
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
```

**Cache invalidation comparison:**

```
❌ BAD ORDERING (Rebuild từ đầu):
COPY . .                      ← Thay code → invalidate
RUN pip install -r requirements.txt  ← Phải rebuild

✅ GOOD ORDERING (Dùng lại cache):
COPY requirements.txt .       ← Ít thay đổi
RUN pip install -r requirements.txt  ← Cache hit!
COPY . .                      ← Thay code (nhanh)
```

**5. Use Specific Version Tags**

```dockerfile
# ❌ BAD: Không xác định
FROM python
FROM pytorch

# ✅ GOOD: Xác định rõ
FROM python:3.9.18-slim
FROM pytorch/pytorch:2.0.1-cuda11.8-runtime-ubuntu20.04
```

**6. Optimize Python Dependencies**

```dockerfile
FROM python:3.9-slim

# ❌ BAD: Cache được lưu
RUN pip install numpy pandas scikit-learn

# ✅ GOOD: Xóa cache
RUN pip install --no-cache-dir numpy pandas scikit-learn

# ✅ BETTER: Multi-line, single RUN
RUN pip install --no-cache-dir \
    numpy==1.24.0 \
    pandas==2.0.0 \
    scikit-learn==1.3.0 \
    && rm -rf /root/.cache/pip
```

**7. Remove Unnecessary Files**

```dockerfile
FROM python:3.9-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# ❌ BAD: Giữ lại build artifacts
# ✅ GOOD: Xóa không cần thiết
RUN apt-get install -y build-essential && \
    pip install numpy && \
    apt-get purge -y build-essential && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*
```

**8. Complete Optimized ML Dockerfile**

```dockerfile
# Stage 1: Builder
FROM pytorch:2.0-cuda11.8 AS builder
WORKDIR /build

# Copy deps first (cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

# Build model
COPY src/ ./src/
COPY data/ ./data/
RUN python src/train.py

# Stage 2: Runtime (final)
FROM python:3.9-slim

# Use non-root user
RUN useradd -m -u 1000 appuser
USER appuser

WORKDIR /app

# Copy only what's needed
COPY --from=builder /build/model.pkl ./models/

# Runtime deps only
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Copy application
COPY src/api.py .

EXPOSE 8000
CMD ["python", "api.py"]
```

**9. Size Comparison**

```bash
# Check image size
$ docker images my-model

REPOSITORY  TAG  SIZE
my-model   v1   500MB    # Unoptimized
my-model   v2   150MB    # With optimizations
my-model   v3   80MB     # Multi-stage + slim

# Detailed layer sizes
$ docker history my-model:v1
```

**10. Optimization Checklist**

- [ ] Base image tối thiểu (slim/alpine nếu possible)
- [ ] Multi-stage builds (tách build/runtime)
- [ ] Layer caching: dependencies trước code
- [ ] .dockerignore: loại bỏ unnecessary files
- [ ] Specific version tags (reproducible)
- [ ] --no-cache-dir cho pip
- [ ] Combine RUN commands (giảm layers)
- [ ] Remove build-only dependencies
- [ ] Non-root user
- [ ] Health checks

**Real-world Result:**

```
Before optimization: 2.3GB
After optimization:  380MB (6x nhỏ hơn!)

Deploy time: 45s → 8s
Storage cost: -74%
Network bandwidth: -74%
```

**Key points:**
- Sử dụng smaller base images (alpine, slim)
- Layer caching strategies
- .dockerignore để exclude unnecessary files
- Multi-stage builds
- Sử dụng specific version tags

### 7. Best practices khi containerize một ML model?

**Câu trả lời:**

Best practices giúp tạo production-ready containers vừa secure vừa efficient.

**Complete Best Practices Dockerfile:**

```dockerfile
# Stage 1: Builder (để compile dependencies)
FROM python:3.9-slim AS builder

WORKDIR /build

# Pin versions
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime (final image)
FROM python:3.9-slim

# Set metadata
LABEL maintainer="ml-team@company.com"
LABEL version="1.0"
LABEL description="ML Model Serving API"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH=/home/mluser/.local/bin:$PATH \
    MODEL_PATH=/app/models \
    LOG_LEVEL=INFO

# Create non-root user
RUN useradd -m -u 1000 mluser && \
    mkdir -p /app/models /app/logs && \
    chown -R mluser:mluser /app

WORKDIR /app

# Copy only dependencies (not full site-packages)
COPY --from=builder --chown=mluser:mluser /root/.local /home/mluser/.local

# Copy application files
COPY --chown=mluser:mluser requirements.txt .
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser model/ ./models/

# Switch to non-root user
USER mluser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run application
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
```

**1. Pin Dependencies Versions**

```dockerfile
# ❌ BAD: Version mismatch across builds
COPY requirements.txt .
RUN pip install -r requirements.txt

# ✅ GOOD: Exact versions for reproducibility
# requirements.txt:
numpy==1.24.3
pandas==2.0.2
scikit-learn==1.3.0
fastapi==0.104.1
uvicorn==0.24.0
```

**2. Multi-stage Build Pattern**

```dockerfile
# Mỗi stage có purpose riêng
FROM pytorch:2.0 AS feature_engineering
# Xử lý dữ liệu, có thể xóa sau

FROM python:3.9-slim AS model_training
# Training, output model.pkl

FROM python:3.9-slim AS runtime
# Only model + serving code
```

**3. Non-root User (Security)**

```dockerfile
# ❌ DANGEROUS: Running as root
FROM ubuntu
ENTRYPOINT ["python", "app.py"]

# ✅ SECURE: Non-root user
FROM ubuntu
RUN useradd -m -u 1000 appuser
USER appuser
ENTRYPOINT ["python", "app.py"]
```

**4. Health Checks**

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1
```

```bash
# Kiểm tra health status
$ docker ps
STATUS              0.0.0.0:8000->8000/tcp (healthy)
$ docker ps
STATUS              0.0.0.0:8000->8000/tcp (unhealthy)
```

**5. Logging Configuration**

```dockerfile
# ✅ Structured logging
ENV LOG_LEVEL=INFO \
    LOG_FORMAT=json

COPY logging_config.yaml /app/
```

```python
# src/api.py
import logging
import json_logging

json_logging.init_non_web(enable_json=True)

logger = logging.getLogger(__name__)
logger.info("Model loaded", extra={"model": "xgboost", "version": "1.0"})
```

**6. Environment Variables**

```dockerfile
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models \
    MODEL_VERSION=1.0 \
    SERVING_PORT=8000 \
    LOG_LEVEL=INFO
```

**7. Volume Mounts (cho models/data)**

```dockerfile
VOLUME ["/data", "/models", "/logs"]
```

```bash
$ docker run \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  my-model:v1
```

**8. Resource Limits**

```bash
$ docker run \
  --memory="2g" \
  --cpus="1.5" \
  --memory-swap="2g" \
  my-model:v1
```

**9. Complete Production Example**

```dockerfile
# Stage 1: Feature Engineering
FROM python:3.9-slim AS features
RUN pip install --no-cache-dir pandas scikit-learn
WORKDIR /features
COPY src/feature_eng.py .
COPY raw_data/ ./data/
RUN python feature_eng.py

# Stage 2: Model Training
FROM pytorch:2.0 AS training
RUN pip install --no-cache-dir torch xgboost optuna
WORKDIR /train
COPY --from=features /features/data ./data/
COPY src/train.py .
RUN python train.py --save /train/model.pkl

# Stage 3: Production Runtime
FROM python:3.9-slim

LABEL maintainer="ml-team@company.com"

# Setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MODEL_PATH=/app/models

# User
RUN useradd -m -u 1000 mluser
USER mluser

WORKDIR /app

# Dependencies
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt

# Model & Code
COPY --from=training /train/model.pkl ./models/
COPY src/api.py .
COPY src/utils.py .

# Metadata
EXPOSE 8000

# Health
HEALTHCHECK --interval=30s \
    CMD curl -f http://localhost:8000/health || exit 1

# Run
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**10. Dockerfile Checklist cho ML Models**

- [ ] **Versioning:** Pin all dependency versions
- [ ] **Multi-stage:** Separate build/runtime stages
- [ ] **Security:** Non-root user, minimal layers
- [ ] **Efficiency:**
  - [ ] Small base image (slim/alpine)
  - [ ] Layer caching (dependencies trước code)
  - [ ] --no-cache-dir for pip
- [ ] **Reliability:**
  - [ ] HEALTHCHECK command
  - [ ] Proper logging
  - [ ] Error handling
- [ ] **Observability:**
  - [ ] ENV variables for configuration
  - [ ] Logging output to stdout/stderr
  - [ ] Metrics exposure
- [ ] **Operations:**
  - [ ] EXPOSE ports
  - [ ] WORKDIR set
  - [ ] ENTRYPOINT/CMD defined
- [ ] **Documentation:**
  - [ ] LABEL metadata
  - [ ] VOLUME definitions
  - [ ] Comments in Dockerfile

**Real-world Deployment Flow:**

```
Local Development:
$ docker build -t mymodel:dev .
$ docker run -it mymodel:dev python train.py

CI/CD Pipeline:
$ docker build -t mymodel:v1.0 .
$ docker run --rm mymodel:v1.0 pytest tests/
$ docker run --rm mymodel:v1.0 python -m mypy src/
$ docker push myrepo/mymodel:v1.0

Production Deployment:
$ docker run -d \
  --name model-api \
  --memory="2g" \
  --cpus="1" \
  -p 8000:8000 \
  --health-cmd="curl localhost:8000/health" \
  -e LOG_LEVEL=WARNING \
  myrepo/mymodel:v1.0
```

**Key points:**
- Pin dependencies versions
- Health checks
- Logging configuration
- Security considerations

### 8. Giải thích về Docker layer caching và cách tối ưu build time?

**Câu trả lời:**

Docker builds images layer by layer, mỗi instruction tạo ra một layer. Nếu input không thay đổi, Docker dùng cached layer, tăng tốc build time.

**1. How Layer Caching Works**

```dockerfile
FROM python:3.9-slim                    # Layer 1
RUN apt-get update                      # Layer 2
COPY requirements.txt .                 # Layer 3
RUN pip install -r requirements.txt     # Layer 4
COPY src/ ./src/                        # Layer 5
RUN python -m pytest tests/             # Layer 6
```

**Layer caching logic:**
```
Layer 1: ✓ Cache hit (base image unchanged)
Layer 2: ✓ Cache hit (apt-get command same)
Layer 3: ✗ Cache MISS (requirements.txt changed)
Layer 4: ✗ Must rebuild (dependency of Layer 3)
Layer 5: ? Depends on src/ files
Layer 6: ? Depends on Layer 5
```

**2. Optimal Layer Ordering**

```dockerfile
# ❌ BAD: Build time 120s
FROM python:3.9-slim
COPY . /app                          # Layer 2: Changes frequently
WORKDIR /app
RUN pip install -r requirements.txt  # Layer 3: Must rebuild even if only test changed

# ✅ GOOD: Build time 15s (if requirements.txt unchanged)
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .              # Layer 2: Rarely changes
RUN pip install -r requirements.txt  # Layer 3: Cache hit!
COPY . .                             # Layer 4: Changes frequently, but deps cached
```

**3. Cache Invalidation Scenarios**

```dockerfile
# Scenario 1: Only code changed
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt    # ✓ Cache hit (30s saved!)
COPY src/ ./src/                       # ✗ Cache miss (new layer)

# Scenario 2: requirements.txt changed
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt    # ✗ Cache miss (must rebuild)
COPY src/ ./src/

# Scenario 3: Dockerfile command changed (even if files same)
FROM python:3.9-slim
RUN apt-get update && apt-get install -y curl  # ✗ Cache miss (command changed)
```

**4. Build Time Optimization Techniques**

**Technique 1: Frequent changes at end**
```dockerfile
# ✅ FAST: Put stable layers first
FROM python:3.9-slim
RUN apt-get update && apt-get install -y build-essential  # Stable
COPY requirements.txt .
RUN pip install -r requirements.txt                       # Relatively stable
COPY src/ ./src/                                          # Changes frequently
COPY config.yaml .                                        # Changes very frequently
```

**Technique 2: Combine RUN commands (reduce layers)**
```dockerfile
# ❌ SLOWER: 3 layers, separate caching
RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y build-essential

# ✅ FASTER: 1 layer, one cache miss = one rebuild
RUN apt-get update && \
    apt-get install -y python3-dev build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

**Technique 3: Use .dockerignore**
```
# .dockerignore
*.pyc
__pycache__
.git
.pytest_cache
tests/
*.zip
node_modules/
.env
```

**Without .dockerignore:**
```
$ docker build .
COPY . /app   # 500MB of files (including .git, cache)
```

**With .dockerignore:**
```
$ docker build .
COPY . /app   # 50MB of files (only code, much faster!)
```

**Technique 4: BuildKit (faster builds)**
```bash
# Enable BuildKit (default in Docker 23+)
DOCKER_BUILDKIT=1 docker build -t myapp .

# BuildKit features:
# - Parallel layer builds
# - Better caching
# - Build secrets
```

**5. Real-world Build Time Comparison**

```
Scenario: Change 1 file in src/model.py

❌ BAD Dockerfile:
FROM python:3.9
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
Build time: 120s (rebuild everything)

✅ GOOD Dockerfile:
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
Build time: 5s (deps cached!)
```

**6. ML Model Build Optimization**

```dockerfile
# ✅ Optimized for ML pipeline
FROM pytorch:2.0 AS builder

WORKDIR /build

# Layer 1: System packages (stable)
RUN apt-get update && \
    apt-get install -y curl git && \
    rm -rf /var/lib/apt/lists/*

# Layer 2: Requirements (changes occasionally)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Layer 3: Code (changes frequently)
COPY src/ ./src/
COPY model/ ./model/

# Layer 4: Training (depends on code/model)
RUN python src/train.py

# Final stage: Runtime (no training tools)
FROM python:3.9-slim
COPY --from=builder /build/model.pkl ./models/
COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt
COPY src/api.py .
```

**7. Cache Busting Strategies**

**When you NEED to invalidate cache:**

```dockerfile
# Option 1: Use --no-cache flag
$ docker build --no-cache -t myapp .

# Option 2: Add a build argument (forces rebuild from here on)
FROM python:3.9-slim
ARG BUILD_DATE
RUN echo "Built on $BUILD_DATE"   # Always different, invalidates cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Build with new date
$ docker build --build-arg BUILD_DATE=$(date) -t myapp .

# Option 3: Change base image tag
FROM python:3.9-slim   # Use exact version, not :latest
# When new 3.9-slim released, manually update to get new layer
```

**8. Inspection & Debugging**

```bash
# See layer sizes and history
$ docker history myapp:v1
IMAGE          CREATED      CREATED BY                         SIZE
abc123         2 hours ago  /bin/sh -c pip install...         450MB
def456         2 hours ago  /bin/sh -c #(nop) COPY...         50MB
ghi789         2 hours ago  /bin/sh -c #(nop) FROM python...  150MB

# See if cache was used
$ DOCKER_BUILDKIT=1 docker build -t myapp .
#10 [stage-1 4/7] COPY requirements.txt .
#10 CACHED

# Deep inspection
$ docker inspect myapp:v1 --format='{{.RootFS.Layers}}'
```

**9. Layer Caching Checklist**

- [ ] **Order:** System deps → Requirements → Code → App-specific
- [ ] **Stable first:** Things that rarely change in earlier layers
- [ ] **Combine:** Use && to combine related RUN commands
- [ ] **.dockerignore:** Exclude unnecessary files
- [ ] **Specific tags:** Pin base image versions (not :latest)
- [ ] **Test builds:** Time your builds before/after optimization
- [ ] **BuildKit:** Enable for parallel builds
- [ ] **Multi-stage:** Separate concerns (build vs runtime)

**10. Build Time Summary Table**

| Optimization | Before | After | Savings |
|-------------|--------|-------|---------|
| Base image only | - | 150MB | - |
| + Multi-stage | 1.5GB | 200MB | 87% |
| + .dockerignore | 200MB | 180MB | 10% |
| + Layer cache | 120s | 5s | 96% |
| + Combine RUN | 8 layers | 5 layers | 37% |
| **Total result** | 1.5GB, 120s | 180MB, 5s | **85% smaller, 24x faster!** |

**Key points:**
- Layer ordering (từ ít thay đổi đến thường xuyên thay đổi)
- COPY requirements trước khi COPY source code
- Build cache invalidation
- Ví dụ về optimal ordering

### 9. Làm thế nào để handle large model files trong Docker?

**Câu trả lời:**

Large model files (hàng GB) là thách thức trong Docker vì:
1. Tăng image size → slow deploy
2. Layer size limits (max ~5GB per layer)
3. Registry storage costs
4. Inefficient to rebuild image when model updates

**Các chiến lược:**

**Strategy 1: Volume Mounting (Development)**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY src/ ./src/
COPY requirements.txt .
RUN pip install -r requirements.txt

# Không COPY model, mount at runtime
EXPOSE 8000
CMD ["python", "src/api.py"]
```

```bash
# Run with volume mount
$ docker run -d \
  -v /path/to/models:/app/models \
  -p 8000:8000 \
  model-api:v1
```

**Advantages:**
- Model updates không cần rebuild
- Dùng lại model cho multiple containers
- Fast development iteration

**Disadvantages:**
- Phục thuộc vào host filesystem
- Không portable

---

**Strategy 2: Download at Runtime (Recommended for Production)**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY src/ ./src/
COPY requirements.txt .
RUN pip install -r requirements.txt

# Đổi entrypoint để download model trước
COPY scripts/download_model.sh .
RUN chmod +x download_model.sh

CMD ["./download_model.sh"]
```

**download_model.sh:**
```bash
#!/bin/bash
set -e

MODEL_URL="s3://ml-models-bucket/xgboost-v1.0.pkl"
MODEL_PATH="/app/models/model.pkl"

mkdir -p /app/models

echo "Downloading model from $MODEL_URL"
aws s3 cp $MODEL_URL $MODEL_PATH

echo "Starting serving..."
exec python src/api.py
```

**Build:**
```bash
$ docker build -t model-api:v1 .
# Image size: 200MB (small!)
```

**Run:**
```bash
$ docker run -d \
  -e AWS_ACCESS_KEY_ID=xxx \
  -e AWS_SECRET_ACCESS_KEY=yyy \
  model-api:v1
# First run downloads model (5 mins)
# Subsequent runs use cached model
```

**Advantages:**
- Portable, no host filesystem dependency
- Model versioning via S3 paths
- Separation of concerns (code vs data)

**Disadvantages:**
- First startup slower (download time)
- Network dependency

---

**Strategy 3: Model Registry (Best Practice)**

Use dedicated model registries like:
- **MLflow Model Registry**
- **BentoML ModelStore**
- **Seldon Model Repository**
- **Azure Model Registry**
- **SageMaker Model Registry**

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install mlflow boto3 -r requirements.txt

COPY src/ ./src/

ENV MLFLOW_REGISTRY_URI=s3://model-registry
ENV MODEL_NAME=xgboost-classifier
ENV MODEL_VERSION=v1.0

CMD ["python", "src/api.py"]
```

**api.py:**
```python
import mlflow.pyfunc
import os

model_name = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION")

# Load from registry
model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

# Start FastAPI
@app.post("/predict")
def predict(data):
    return model.predict(data)
```

---

**Strategy 4: Multi-stage with Model Cache**

```dockerfile
# Stage 1: Download model (cached)
FROM python:3.9-slim AS model_cache

RUN pip install awscli
WORKDIR /models
RUN aws s3 cp s3://ml-bucket/xgboost-v1.0.pkl .

# Stage 2: Runtime
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy only model from cache stage
COPY --from=model_cache /models/xgboost-v1.0.pkl ./models/

COPY src/ ./src/
CMD ["python", "src/api.py"]
```

**Result:**
```
Final image: 250MB (model included, but leveraging cache for fast rebuilds)
```

---

**Strategy 5: Docker Layers with External Storage**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y curl

# Python deps
COPY requirements.txt .
RUN pip install -r requirements.txt

# Application
COPY src/ ./src/

# Model: Reference only, download at runtime
ENV MODEL_PATH=/app/models/model.pkl
ENV MODEL_URL=https://cdn.example.com/models/xgboost-v1.0.pkl

RUN mkdir -p $(dirname $MODEL_PATH) && \
    curl -o $MODEL_PATH $MODEL_URL

EXPOSE 8000
CMD ["python", "src/api.py"]
```

---

**Strategy 6: Comparison Table**

| Strategy | Image Size | Startup | Portability | Updates | Complexity |
|----------|-----------|---------|-------------|---------|-----------|
| **Volume Mount** | 150MB | Instant | ✗ (host dep) | Easy | Low |
| **Download at Runtime** | 150MB | Slow (5m) | ✓ | Easy | Medium |
| **Model Registry** | 150MB | Slow (5m) | ✓ | Easy | High |
| **Multi-stage Cache** | 800MB | Instant | ✓ | Rebuild | Medium |
| **COPY in Dockerfile** | 2000MB | Instant | ✓ | Rebuild | Low |

---

**Strategy 7: Production Architecture**

```
┌─────────────────────────────────────┐
│   Docker Image Registry             │
│  (small, code + runtime only)       │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   Container                         │
│  - Download model from S3 on start  │
│  - Cache in /app/models volume      │
└──────────┬──────────────────────────┘
           │
           ▼
┌─────────────────────────────────────┐
│   Model Storage                     │
│  - AWS S3 / GCS bucket              │
│  - Versioned: model-v1.0.pkl        │
└─────────────────────────────────────┘
```

**Dockerfile:**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY scripts/download_model.sh /app/

RUN chmod +x /app/download_model.sh

# Volume for model caching
VOLUME ["/app/models"]

# Health check
HEALTHCHECK --interval=30s CMD curl http://localhost:8000/health

EXPOSE 8000
ENTRYPOINT ["/app/download_model.sh"]
```

---

**Strategy 8: Real-world Example (MLOps Pipeline)**

```dockerfile
# Multi-model serving with lazy loading
FROM python:3.9-slim

WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code
COPY src/ ./src/

ENV MODEL_REGISTRY_URL=https://models.company.com
ENV CACHE_DIR=/app/models

# Models downloaded on demand (lazy loading)
CMD ["python", "src/model_server.py"]
```

**src/model_server.py:**
```python
import os
import requests
from functools import lru_cache

CACHE_DIR = os.getenv("CACHE_DIR", "/app/models")

@lru_cache(maxsize=10)
def load_model(model_name, version):
    cache_path = f"{CACHE_DIR}/{model_name}-{version}.pkl"

    # Check if cached
    if os.path.exists(cache_path):
        return joblib.load(cache_path)

    # Download if not cached
    url = f"{MODEL_REGISTRY_URL}/{model_name}/{version}"
    response = requests.get(url)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path, 'wb') as f:
        f.write(response.content)

    return joblib.load(cache_path)

@app.post("/predict")
def predict(model_name: str, version: str, data: dict):
    model = load_model(model_name, version)
    return model.predict(data)
```

---

**Strategy 9: Large Model Optimization Checklist**

- [ ] **Don't include models in base image** (unless essential)
- [ ] **Download at runtime** or mount volumes
- [ ] **Use external storage** (S3, GCS, Azure Blob)
- [ ] **Lazy loading** for multiple models
- [ ] **Caching** to avoid repeated downloads
- [ ] **Versioning** for model updates
- [ ] **Health checks** to verify model loaded
- [ ] **Monitoring** download times, failures

**Key points:**
- Volume mounting
- Docker layer size limits
- External storage (S3, GCS) và download at runtime
- Model registry integration
- Multi-stage builds để minimize final image size

### 10. Security best practices khi build Docker images cho ML?

**Câu trả lời:**

Security là critical khi deploy ML models vì:
1. **Data exposure:** Models có thể chứa thông tin từ training data
2. **Model theft:** Competitors có thể đánh cắp valuable models
3. **Injection attacks:** Input data có thể được manipulated
4. **Supply chain:** Dependencies có thể bị compromise

**1. Run as Non-root User**

```dockerfile
# ❌ DANGEROUS: Running as root
FROM python:3.9-slim
COPY src/ ./src/
CMD ["python", "src/api.py"]

# ✅ SECURE: Non-root user
FROM python:3.9-slim

# Create user
RUN useradd -m -u 1000 mluser

WORKDIR /app

# Install as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy files with proper ownership
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser model/ ./model/

# Switch user
USER mluser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Why non-root?**
- If container escaped, attacker limited to non-root permissions
- Prevents modification of system files
- Container best practice

---

**2. Scan for Vulnerabilities**

```bash
# Using Trivy (free, open source)
$ docker build -t myapp .
$ trivy image myapp:latest

Vulnerabilities found:
HIGH   Fixed in: apt-utils 2.4.1
MEDIUM Fixed in: openssl 1.1.1n

# Using Snyk (cloud-based)
$ snyk container test myapp:latest

# Using Grype (Anchore tool)
$ grype myapp:latest
```

**In CI/CD Pipeline:**
```yaml
# GitHub Actions example
- name: Build image
  run: docker build -t myapp .

- name: Scan with Trivy
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: myapp:latest
    format: 'sarif'
    output: 'trivy-results.sarif'

- name: Upload results
  uses: github/codeql-action/upload-sarif@v2
  with:
    sarif_file: 'trivy-results.sarif'
```

---

**3. Use Official Base Images**

```dockerfile
# ❌ RISKY: Unknown/unofficial
FROM mycompany/python
FROM ubuntu

# ✅ SAFE: Official Docker images
FROM python:3.9-slim
FROM ubuntu:22.04

# Pin specific versions
FROM python:3.9.18-slim-bookworm
FROM ubuntu:22.04

# Verify image signature
$ docker pull python:3.9-slim --disable-content-trust=false
```

**Official Images Checklist:**
- [ ] From official Docker Hub repository
- [ ] Has verification checksum
- [ ] Recent updates for security patches
- [ ] Active maintenance

---

**4. Secret Management (Never Hard-code Credentials)**

```dockerfile
# ❌ TERRIBLE: Secrets in Dockerfile
FROM python:3.9-slim
ENV AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
ENV AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
COPY src/ ./src/
```

```dockerfile
# ✅ GOOD: Use secrets at runtime
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ ./src/

# Don't set secrets here!
CMD ["python", "src/api.py"]
```

```bash
# Run with secrets from external source
$ docker run -d \
  -e AWS_ACCESS_KEY_ID=$(aws sts get-caller-identity) \
  -e AWS_SECRET_ACCESS_KEY=$(get-secret-from-vault) \
  myapp:latest
```

**Using Docker Secrets (Swarm):**
```dockerfile
# Create secret
$ echo "super-secret-key" | docker secret create api_key -

# Reference in compose
version: '3.1'
services:
  api:
    image: myapp
    secrets:
      - api_key
    environment:
      API_KEY_FILE: /run/secrets/api_key

secrets:
  api_key:
    external: true
```

---

**5. Minimize Attack Surface**

```dockerfile
# ❌ BAD: Include unnecessary packages
FROM ubuntu
RUN apt-get update && \
    apt-get install -y build-essential curl wget git docker.io

# ✅ GOOD: Only what's needed
FROM python:3.9-slim
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# ✅ BETTER: Use distroless images (bare minimum)
FROM gcr.io/distroless/python3
COPY src/ ./src/
ENTRYPOINT ["python", "src/api.py"]
```

**Image sizes comparison:**
```
ubuntu:latest          77MB
python:3.9            886MB
python:3.9-slim       151MB
python:3.9-alpine      50MB
distroless/python3     30MB  ← Minimal!
```

---

**6. Regular Updates and Patch Management**

```dockerfile
# ❌ BAD: Pinned forever, no updates
FROM python:3.9-slim

# ✅ GOOD: Regular rebuild schedule
# Rebuild weekly in CI/CD to get security patches
```

```yaml
# GitHub Actions: Rebuild weekly
name: Security Rebuild
on:
  schedule:
    - cron: '0 0 * * 0'  # Weekly

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build and push
        run: |
          docker build -t myapp:latest .
          docker push myapp:latest
```

---

**7. File Permissions and Ownership**

```dockerfile
# ✅ CORRECT: Proper file ownership
FROM python:3.9-slim

RUN useradd -m -u 1000 mluser

WORKDIR /app

COPY --chown=mluser:mluser requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=mluser:mluser src/ ./src/

# Ensure no world-writable files
RUN find /app -type f -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

USER mluser
```

---

**8. Content Trust and Image Signing**

```bash
# Enable content trust
$ export DOCKER_CONTENT_TRUST=1

# Sign and push image
$ docker push myregistry/myapp:v1.0
# Requires signing key

# Verify signed image
$ docker pull myregistry/myapp:v1.0
# Checks signature before pull
```

---

**9. Read-only Filesystem**

```bash
# Run container with read-only filesystem
$ docker run --read-only \
  --tmpfs /tmp:rw \
  --tmpfs /app/logs:rw \
  myapp:latest
```

```dockerfile
# In Dockerfile, prepare writable directories
FROM python:3.9-slim

WORKDIR /app

COPY src/ ./src/

# Create directories for runtime writes
RUN mkdir -p /tmp /app/logs && \
    chmod 1777 /tmp

USER 1000
```

---

**10. Complete Secure Dockerfile Example**

```dockerfile
# Stage 1: Builder
FROM python:3.9-slim AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime (final)
FROM python:3.9-slim

# Metadata
LABEL maintainer="ml-team@company.com"
LABEL version="1.0"

# Create non-root user
RUN groupadd -r mluser && \
    useradd -r -g mluser -u 1000 mluser

# Set umask for secure file creation
RUN umask 0027

WORKDIR /app

# Copy dependencies
COPY --from=builder --chown=mluser:mluser /root/.local /home/mluser/.local

# Copy application (read-only)
COPY --chown=mluser:mluser --chmod=555 src/ ./src/
COPY --chown=mluser:mluser --chmod=555 model/ ./model/

# Create writable directories for runtime
RUN mkdir -p /tmp /app/logs && \
    chown mluser:mluser /tmp /app/logs && \
    chmod 1777 /tmp && \
    chmod 755 /app/logs

# Set environment
ENV PYTHONUNBUFFERED=1 \
    PATH=/home/mluser/.local/bin:$PATH

# Security headers
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONFAULTHANDLER=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s \
    CMD curl -f http://localhost:8000/health || exit 1

# Non-root user
USER mluser

EXPOSE 8000

# Run as exec form (receives signals properly)
CMD ["python", "-u", "src/api.py"]
```

---

**11. Security Checklist**

- [ ] **Non-root user** (USER statement)
- [ ] **Scan vulnerabilities** (Trivy, Snyk)
- [ ] **Official base images** (verified, updated)
- [ ] **No hardcoded secrets** (env vars at runtime)
- [ ] **Minimal base image** (slim/alpine/distroless)
- [ ] **Updated dependencies** (security patches)
- [ ] **File permissions** (644 for files, 755 for dirs)
- [ ] **Read-only filesystem** (where possible)
- [ ] **Image signing** (content trust)
- [ ] **Regular rebuilds** (weekly for patches)
- [ ] **Health checks** (verify container health)
- [ ] **Logging** (for audit trail)
- [ ] **SBOM** (Software Bill of Materials)

---

**12. Real-world Security Flow**

```
Developer writes code
    ↓
Build Docker image
    ↓
Scan with Trivy → Fix vulnerabilities
    ↓
Sign image
    ↓
Push to private registry
    ↓
Deploy with read-only FS
    ↓
Monitor logs & metrics
    ↓
Weekly security rebuild
```

**Key points:**
- Run as non-root user
- Scan for vulnerabilities (Trivy, Snyk)
- Minimize attack surface
- Secret management (không hard-code credentials)
- Use official base images
- Regular image updates
- Minimal base images

---

[Tiếp theo: Docker Compose →](./02-docker-compose.md)
