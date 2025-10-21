# Docker & Dockerfile

[← Quay lại](./README.md)

## Câu hỏi cơ bản

### 1. Docker là gì và tại sao nó quan trọng trong MLOps?

**Key points:**
- Giải thích về containerization
- Lợi ích: reproducibility, isolation, portability
- Vai trò trong ML workflow

### 2. Sự khác biệt giữa Docker Image và Docker Container?

**Key points:**
- Image: template, immutable
- Container: running instance của image
- Lifecycle và quản lý

### 3. Giải thích cấu trúc của một Dockerfile?

**Key points:**
- Base image (FROM)
- Working directory (WORKDIR)
- Copy files (COPY, ADD)
- Install dependencies (RUN)
- Environment variables (ENV)
- Entrypoint và CMD

### 4. Phân biệt CMD và ENTRYPOINT trong Dockerfile?

**Key points:**
- CMD: default command, có thể override
- ENTRYPOINT: main command, append arguments
- Best practices khi kết hợp cả hai

### 5. Multi-stage builds trong Docker là gì? Tại sao sử dụng?

**Key points:**
- Giảm kích thước image
- Tách build và runtime dependencies
- Ví dụ: compile model trong stage 1, deploy trong stage 2

---

## Câu hỏi nâng cao

### 6. Làm thế nào để optimize Docker image cho ML models?

**Key points:**
- Sử dụng smaller base images (alpine, slim)
- Layer caching strategies
- .dockerignore để exclude unnecessary files
- Multi-stage builds
- Sử dụng specific version tags

### 7. Best practices khi containerize một ML model?

**Example Dockerfile:**
```dockerfile
# Example structure
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model/ ./model/
COPY src/ ./src/
ENV MODEL_PATH=/app/model/model.pkl
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Best practices:**
- Pin dependencies versions
- Health checks
- Logging configuration
- Security considerations

### 8. Giải thích về Docker layer caching và cách tối ưu build time?

**Key points:**
- Layer ordering (từ ít thay đổi đến thường xuyên thay đổi)
- COPY requirements trước khi COPY source code
- Build cache invalidation
- Ví dụ về optimal ordering

### 9. Làm thế nào để handle large model files trong Docker?

**Strategies:**
- Volume mounting
- Docker layer size limits
- External storage (S3, GCS) và download at runtime
- Model registry integration
- Multi-stage builds để minimize final image size

### 10. Security best practices khi build Docker images cho ML?

**Key points:**
- Run as non-root user
- Scan for vulnerabilities (Trivy, Snyk)
- Minimize attack surface
- Secret management (không hard-code credentials)
- Use official base images
- Regular image updates
- Minimal base images

**Example secure Dockerfile:**
```dockerfile
FROM python:3.9-slim

# Create non-root user
RUN useradd -m -u 1000 mluser

WORKDIR /app

# Copy and install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=mluser:mluser src/ ./src/
COPY --chown=mluser:mluser model/ ./model/

# Switch to non-root user
USER mluser

EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

[Tiếp theo: Docker Compose →](./02-docker-compose.md)
