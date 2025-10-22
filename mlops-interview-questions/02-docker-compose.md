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

**Giải thích chi tiết:**

Docker Compose là công cụ cho phép định nghĩa và chạy nhiều Docker containers như một ứng dụng duy nhất. Thay vì chạy từng container riêng lẻ bằng lệnh `docker run`, bạn viết một file `docker-compose.yml` định nghĩa tất cả các services cần thiết.

**Cách hoạt động:**
- Compose đọc file YAML, xử lý các biến môi trường, xây dựng images (nếu cần) và khởi động containers
- Tất cả containers được kết nối thông qua một network mặc định, cho phép chúng giao tiếp với nhau qua service names
- Dữ liệu được quản lý thông qua volumes

**Khi nào sử dụng Compose:**
- **Development environments**: Developers có thể chạy toàn bộ stack ứng dụng chỉ bằng `docker-compose up`
- **Testing & CI/CD**: Tạo isolated test environments một cách nhanh chóng
- **Microservices nhỏ**: Trong giai đoạn phát triển, Compose đủ để quản lý các services đơn giản
- **Local development**: Thay thế cho việc cài đặt nhiều dependencies trên máy local

**Compose vs Kubernetes:**
- **Compose**: Dùng cho development, testing, single-host deployments. Đơn giản, dễ học
- **Kubernetes**: Production-grade orchestration, multi-host, auto-scaling, rolling updates, advanced networking

### 12. Cấu trúc cơ bản của docker-compose.yml?

**Example chi tiết:**
```yaml
version: '3.8'  # Version định danh features khả dụng

services:       # Các containers sẽ chạy
  api:
    build:      # Build image từ Dockerfile
      context: ./api
      dockerfile: Dockerfile
    image: my-api:1.0              # Hoặc dùng image sẵn có
    container_name: api-container  # Tên container
    ports:
      - "8000:8000"               # host:container port mapping
    environment:                   # Biến môi trường
      - DATABASE_URL=postgresql://user:pass@db:5432/mydb
      - DEBUG=true
    env_file:
      - .env                       # Load từ file .env
    volumes:
      - ./src:/app/src            # Bind mount (code reload)
      - app-data:/app/data        # Named volume (persistence)
    depends_on:
      - db                        # Thứ tự khởi động
    networks:
      - backend                   # Kết nối tới network
    restart: unless-stopped       # Restart policy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  db:
    image: postgres:13
    environment:
      POSTGRES_DB: mydb
      POSTGRES_PASSWORD: secret
    volumes:
      - postgres-data:/var/lib/postgresql/data
    networks:
      - backend

volumes:        # Named volumes - tồn tại độc lập
  postgres-data:
  app-data:

networks:       # Custom networks
  backend:
    driver: bridge
```

**Giải thích từng thành phần:**
- **version**: Mỗi version hỗ trợ các features khác nhau (3.8 là khá mới)
- **services**: Định nghĩa các container cần chạy
- **volumes**: Khai báo volumes để tái sử dụng hoặc lưu trữ dữ liệu
- **networks**: Tạo networks tùy chỉnh để kiểm soát giao tiếp giữa containers
- **environment**: Biến môi trường được truyền vào container
- **ports**: Map port từ host sang container
- **depends_on**: Định nghĩa thứ tự khởi động containers
- **restart**: Chính sách restart khi container dừng
- **healthcheck**: Kiểm tra service có sẵn sàng không

### 13. Giải thích về networking trong Docker Compose?

**Key points:**
- Default bridge network
- Custom networks
- Service discovery by service name
- Port mapping (host:container)
- Network isolation

**Giải thích chi tiết:**

Mặc định, Docker Compose tạo một **bridge network** cho các services có thể giao tiếp với nhau.

**Cơ chế service discovery:**
```yaml
services:
  web:
    build: .
    ports:
      - "8000:8000"      # Expose ra host machine

  database:
    image: postgres:13
    # Không cần expose, chỉ giao tiếp nội bộ
```

- Service `web` có thể truy cập `database` bằng hostname `database` (tên service)
- Compose tự động tạo DNS entries cho mỗi service
- Connection string trong `web` có thể là: `postgresql://user:pass@database:5432/db`

**Custom Networks - Network Isolation:**
```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge

services:
  web:
    networks:
      - frontend      # Chỉ kết nối tới frontend network

  api:
    networks:
      - frontend      # Kết nối tới cả hai
      - backend

  db:
    networks:
      - backend       # Chỉ backend có thể truy cập
```

Lợi ích: `web` không thể trực tiếp truy cập `db` vì chúng không chia sẻ network

**Port mapping:**
- `"8000:8000"`: host port 8000 → container port 8000
- Chỉ port mapping nào được expose ra ngoài, service names chỉ hoạt động nội bộ
- Ví dụ: `web` từ ngoài máy truy cập `localhost:8000`, nhưng `api` truy cập `web` bằng `http://web:8000`

### 14. Volumes trong Docker Compose hoạt động như thế nào?

**Key points:**
- Named volumes
- Bind mounts
- Data persistence
- Sharing data giữa containers
- Volume drivers

**Giải thích chi tiết:**

**Loại 1: Named Volumes** (Managed by Docker)
```yaml
volumes:
  db-data:        # Docker tự quản lý vị trí lưu trữ

services:
  db:
    volumes:
      - db-data:/var/lib/postgresql/data
```
- Docker lưu trữ ở `/var/lib/docker/volumes/db-data/_data`
- Tồn tại độc lập với containers
- Dễ backup, share giữa containers
- Độ ưu tiên: cao (recommended cho production)

**Loại 2: Bind Mounts** (Host directories)
```yaml
services:
  api:
    volumes:
      - ./src:/app/src        # Host path:Container path
      - ./config:/app/config
```
- Trực tiếp map thư mục host vào container
- Thay đổi trên host tự động phản ánh trong container (useful cho development)
- Không quản lý bởi Docker

**Loại 3: Anonymous Volumes**
```yaml
services:
  app:
    volumes:
      - /app/data     # Chỉ định path container
```
- Docker tự tạo named volume ẩn danh
- Sử dụng hiếm

**Data persistence example:**
```yaml
volumes:
  postgres-data:
  redis-data:

services:
  db:
    image: postgres:13
    volumes:
      - postgres-data:/var/lib/postgresql/data
    environment:
      POSTGRES_PASSWORD: secret

  cache:
    image: redis:7
    volumes:
      - redis-data:/data
```

Khi chạy `docker-compose down`, volumes vẫn tồn tại. Dữ liệu được bảo tồn cho lần khởi động tiếp theo.

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

**Phương pháp 1: Direct environment section**
```yaml
services:
  api:
    environment:
      - DEBUG=true
      - LOG_LEVEL=info
      - DATABASE_TIMEOUT=30
```

**Phương pháp 2: Biến môi trường từ host**
```yaml
# docker-compose.yml
services:
  api:
    environment:
      - DATABASE_PASSWORD=${DB_PASSWORD}  # Sẽ lấy từ shell env
      - API_KEY=${API_KEY}
```

Chạy:
```bash
export DB_PASSWORD=secret123
export API_KEY=key456
docker-compose up
```

**Phương pháp 3: .env file**
```bash
# .env (giống environment)
DB_PASSWORD=secret123
API_KEY=key456
DATABASE_URL=postgresql://user:pass@db:5432/mydb
```

```yaml
# docker-compose.yml
services:
  api:
    env_file:
      - .env              # Load tất cả biến từ .env
```

Compose tự động load từ `.env` nếu file tồn tại

**Phương pháp 4: env_file từ multiple files**
```yaml
services:
  api:
    env_file:
      - .env                    # Base config
      - .env.local              # Local overrides
      - .env.${ENVIRONMENT}     # Environment-specific
```

**Phương pháp 5: Secrets (Docker Swarm mode)**
```yaml
services:
  api:
    secrets:
      - db_password
      - api_key
    environment:
      - DB_PASSWORD_FILE=/run/secrets/db_password

secrets:
  db_password:
    file: ./secrets/db_password.txt
  api_key:
    file: ./secrets/api_key.txt
```

**Best Practices:**
```bash
# .env.example - Commit vào git
DB_PASSWORD=change_me
API_KEY=change_me

# .env - KHÔNG commit vào git (.gitignore)
DB_PASSWORD=actual_secret
API_KEY=actual_secret
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  database:
    image: postgres:13
    env_file: .env
    environment:
      - POSTGRES_PASSWORD=${DB_PASSWORD}

  api:
    build: .
    env_file:
      - .env
      - .env.local
    environment:
      - DATABASE_URL=postgresql://user:${DB_PASSWORD}@database:5432/mydb
      - FLASK_ENV=${ENVIRONMENT:-development}
```

**Order of precedence** (từ cao tới thấp):
1. `environment` section trong compose
2. `env_file` list
3. Biến từ shell (.env tự động load)

### 17. Health checks và dependencies trong Docker Compose?

**Key points:**
- healthcheck directive
- depends_on với conditions
- Startup order và wait-for scripts
- Restart policies

**Vấn đề:** Chỉ `depends_on` không đảm bảo service sẵn sàng

```yaml
# ❌ KHÔNG an toàn - service có thể chưa ready
services:
  api:
    depends_on:
      - db    # db container chạy nhưng PostgreSQL chưa ready

  db:
    image: postgres:13
```

Thường phải chờ PostgreSQL khởi động và accept connections (có thể 5-10 giây)

**Giải pháp: Health Checks**

```yaml
services:
  database:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: secret
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres -h localhost"]
      interval: 10s       # Kiểm tra mỗi 10 giây
      timeout: 5s         # Timeout 5 giây
      retries: 5          # Thất bại sau 5 lần không thành công
      start_period: 0s    # Không chờ trước khi bắt đầu kiểm tra
    networks:
      - backend

  api:
    build: .
    depends_on:
      database:
        condition: service_healthy    # Chờ database healthy
    networks:
      - backend
```

**Các loại healthcheck:**

```yaml
# Loại 1: Shell command
healthcheck:
  test: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"]

# Loại 2: Exec format
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/health"]

# Loại 3: Bash script
healthcheck:
  test: ["CMD-SHELL", "if [ -f /app/ready ]; then exit 0; else exit 1; fi"]

# Loại 4: Vô hiệu hóa healthcheck
healthcheck:
  disable: true
```

**Complete ML Pipeline example với health checks:**

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:13
    environment:
      POSTGRES_PASSWORD: mlflow_pass
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - ml-network

  mlflow:
    image: ghcr.io/mlflow/mlflow:latest
    depends_on:
      postgres:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000"]
      interval: 15s
      timeout: 5s
      retries: 3
    networks:
      - ml-network
    command: mlflow server --host 0.0.0.0

  training:
    build: ./training
    depends_on:
      mlflow:
        condition: service_healthy
    environment:
      MLFLOW_TRACKING_URI: http://mlflow:5000
    networks:
      - ml-network

  api:
    build: ./api
    ports:
      - "8000:8000"
    depends_on:
      - training
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s    # Chờ 40s trước khi kiểm tra (thời gian startup)
    networks:
      - ml-network

  prometheus:
    image: prom/prometheus:latest
    depends_on:
      - api
    networks:
      - ml-network

networks:
  ml-network:
    driver: bridge
```

**Restart Policies:**

```yaml
services:
  api:
    restart: no              # Không tự động restart

  worker:
    restart: always          # Luôn restart nếu dừng

  cache:
    restart: on-failure      # Restart chỉ khi exit với error

  db:
    restart: unless-stopped  # Restart trừ khi bị stop thủ công
```

**Các trạng thái của service:**
- `starting`: Container đang khởi động
- `healthy`: Healthcheck thành công
- `unhealthy`: Healthcheck thất bại
- `exited`: Container đã dừng
- `created`: Container được tạo nhưng chưa chạy

---

[← Docker & Dockerfile](./01-docker-dockerfile.md) | [CI/CD →](./03-cicd.md)
