# Bộ Câu Hỏi Phỏng Vấn MLOps

## Mục lục
1. [Docker & Dockerfile](#docker--dockerfile)
2. [Docker Compose](#docker-compose)
3. [CI/CD](#cicd)
4. [Kubernetes](#kubernetes)
5. [MLOps Tools & Practices](#mlops-tools--practices)
6. [Câu hỏi tình huống](#câu-hỏi-tình-huống)

---

## Docker & Dockerfile

### Câu hỏi cơ bản

**1. Docker là gì và tại sao nó quan trọng trong MLOps?**
- Giải thích về containerization
- Lợi ích: reproducibility, isolation, portability
- Vai trò trong ML workflow

**2. Sự khác biệt giữa Docker Image và Docker Container?**
- Image: template, immutable
- Container: running instance của image
- Lifecycle và quản lý

**3. Giải thích cấu trúc của một Dockerfile?**
- Base image (FROM)
- Working directory (WORKDIR)
- Copy files (COPY, ADD)
- Install dependencies (RUN)
- Environment variables (ENV)
- Entrypoint và CMD

**4. Phân biệt CMD và ENTRYPOINT trong Dockerfile?**
- CMD: default command, có thể override
- ENTRYPOINT: main command, append arguments
- Best practices khi kết hợp cả hai

**5. Multi-stage builds trong Docker là gì? Tại sao sử dụng?**
- Giảm kích thước image
- Tách build và runtime dependencies
- Ví dụ: compile model trong stage 1, deploy trong stage 2

### Câu hỏi nâng cao

**6. Làm thế nào để optimize Docker image cho ML models?**
- Sử dụng smaller base images (alpine, slim)
- Layer caching strategies
- .dockerignore để exclude unnecessary files
- Multi-stage builds
- Sử dụng specific version tags

**7. Best practices khi containerize một ML model?**
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
- Pin dependencies versions
- Health checks
- Logging configuration
- Security considerations

**8. Giải thích về Docker layer caching và cách tối ưu build time?**
- Layer ordering (từ ít thay đổi đến thường xuyên thay đổi)
- COPY requirements trước khi COPY source code
- Build cache invalidation

**9. Làm thế nào để handle large model files trong Docker?**
- Volume mounting
- Docker layer size limits
- External storage (S3, GCS) và download at runtime
- Model registry integration

**10. Security best practices khi build Docker images cho ML?**
- Run as non-root user
- Scan for vulnerabilities
- Minimize attack surface
- Secret management (không hard-code credentials)
- Use official base images

---

## Docker Compose

### Câu hỏi cơ bản

**11. Docker Compose là gì và khi nào nên sử dụng?**
- Multi-container applications
- Development và testing environments
- Service orchestration cơ bản

**12. Cấu trúc cơ bản của docker-compose.yml?**
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

**13. Giải thích về networking trong Docker Compose?**
- Default bridge network
- Custom networks
- Service discovery by service name
- Port mapping

**14. Volumes trong Docker Compose hoạt động như thế nào?**
- Named volumes
- Bind mounts
- Data persistence
- Sharing data giữa containers

### Câu hỏi nâng cao

**15. Thiết kế docker-compose.yml cho ML pipeline với các components:**
- Model training service
- Model serving API
- Database (PostgreSQL)
- Model registry (MLflow)
- Monitoring (Prometheus + Grafana)

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

volumes:
  mlflow-data:
  db-data:
```

**16. Environment variables và secrets management trong Docker Compose?**
- .env files
- environment section
- env_file directive
- Docker secrets (swarm mode)

**17. Health checks và dependencies trong Docker Compose?**
- healthcheck directive
- depends_on với conditions
- Startup order và wait-for scripts

---

## CI/CD

### Câu hỏi cơ bản

**18. CI/CD là gì và tại sao quan trọng trong MLOps?**
- Continuous Integration: automated testing, building
- Continuous Deployment: automated deployment
- Benefits: faster iteration, reproducibility, quality assurance

**19. Các stages chính trong ML CI/CD pipeline?**
- Code quality checks (linting, formatting)
- Unit tests
- Data validation
- Model training
- Model evaluation
- Model testing (integration, performance)
- Model deployment
- Monitoring

**20. Sự khác biệt giữa traditional CI/CD và ML CI/CD?**
- Data versioning
- Model versioning
- Data validation stages
- Model evaluation gates
- A/B testing và gradual rollouts
- Model monitoring

### Câu hỏi nâng cao

**21. Thiết kế một complete CI/CD pipeline cho ML project (GitHub Actions)?**

```yaml
# .github/workflows/ml-pipeline.yml
name: ML Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install flake8 black isort
      - name: Lint with flake8
        run: flake8 src/ tests/
      - name: Check formatting
        run: black --check src/ tests/
      - name: Check imports
        run: isort --check-only src/ tests/

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest tests/ --cov=src --cov-report=xml

  data-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Validate data schema
        run: |
          python scripts/validate_data.py
      - name: Check data quality
        run: |
          python scripts/data_quality_checks.py

  train-model:
    needs: [code-quality, unit-tests, data-validation]
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
      - name: Train model
        run: python src/train.py
      - name: Evaluate model
        run: python src/evaluate.py
      - name: Upload model artifacts
        uses: actions/upload-artifact@v3
        with:
          name: model
          path: models/

  build-and-push:
    needs: train-model
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: docker build -t mymodel:${{ github.sha }} .
      - name: Push to registry
        run: |
          echo ${{ secrets.DOCKER_PASSWORD }} | docker login -u ${{ secrets.DOCKER_USERNAME }} --password-stdin
          docker push mymodel:${{ github.sha }}

  deploy-staging:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/develop'
    steps:
      - name: Deploy to staging
        run: |
          kubectl set image deployment/model-api api=mymodel:${{ github.sha }} -n staging

  deploy-production:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production
    steps:
      - name: Deploy to production
        run: |
          kubectl set image deployment/model-api api=mymodel:${{ github.sha }} -n production
```

**22. Continuous Training (CT) pipeline design?**
- Trigger mechanisms (schedule, data drift, performance degradation)
- Automated retraining workflow
- Model comparison và approval gates
- Automated deployment của better models

**23. Model testing strategies trong CI/CD?**
- Unit tests for preprocessing functions
- Integration tests for full pipeline
- Model performance tests (accuracy thresholds)
- Inference time benchmarks
- Data drift tests
- Model bias tests
- Backward compatibility tests

**24. GitOps cho ML deployments?**
- Infrastructure as Code
- Declarative deployments
- Version control for all configs
- Automated sync between Git và cluster state

**25. Các công cụ CI/CD phổ biến cho MLOps?**
- GitHub Actions
- GitLab CI/CD
- Jenkins
- CircleCI
- Azure DevOps
- Argo Workflows
- Kubeflow Pipelines

---

## Kubernetes

### Câu hỏi cơ bản

**26. Kubernetes là gì và tại sao dùng cho MLOps?**
- Container orchestration platform
- Auto-scaling, self-healing, load balancing
- Resource management cho training và serving
- Multi-environment support

**27. Các components chính của Kubernetes?**
- Control Plane: API Server, Scheduler, Controller Manager, etcd
- Worker Nodes: Kubelet, Kube-proxy, Container Runtime
- Pods, Services, Deployments

**28. Pod là gì? Deployment là gì?**
- Pod: smallest deployable unit, 1+ containers
- Deployment: manages ReplicaSets, declarative updates
- Scaling và rolling updates

**29. Services trong Kubernetes và các types?**
- ClusterIP: internal communication
- NodePort: external access via node port
- LoadBalancer: cloud load balancer
- Service discovery

**30. ConfigMaps và Secrets?**
- ConfigMaps: non-sensitive configuration
- Secrets: sensitive data (base64 encoded)
- Mounting as volumes or environment variables

### Câu hỏi nâng cao

**31. Deploy một ML model trên Kubernetes - các resources cần thiết?**

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-serving
  labels:
    app: model-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: model-api
  template:
    metadata:
      labels:
        app: model-api
    spec:
      containers:
      - name: api
        image: mymodel:v1.0.0
        ports:
        - containerPort: 8000
        env:
        - name: MODEL_PATH
          value: /models/model.pkl
        - name: WORKERS
          value: "4"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: model-storage
          mountPath: /models
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer

---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-serving
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**32. Resource management cho ML workloads?**
- Requests vs Limits
- GPU resources (nvidia.com/gpu)
- Memory-intensive training jobs
- Resource quotas và limit ranges
- Node affinity cho GPU nodes

**33. Training jobs trên Kubernetes - Job vs CronJob?**

```yaml
# training-job.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: model-training
spec:
  template:
    spec:
      containers:
      - name: trainer
        image: training-image:v1
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 16Gi
            cpu: 4
        env:
        - name: MLFLOW_TRACKING_URI
          valueFrom:
            configMapKeyRef:
              name: mlflow-config
              key: tracking_uri
        volumeMounts:
        - name: data
          mountPath: /data
      restartPolicy: Never
      volumes:
      - name: data
        persistentVolumeClaim:
          claimName: training-data-pvc
  backoffLimit: 3

---
# Scheduled retraining
apiVersion: batch/v1
kind: CronJob
metadata:
  name: weekly-retraining
spec:
  schedule: "0 2 * * 0"  # 2 AM every Sunday
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: trainer
            image: training-image:v1
            # ... same as above
          restartPolicy: OnFailure
```

**34. Persistent storage cho ML trong Kubernetes?**
- PersistentVolume (PV) và PersistentVolumeClaim (PVC)
- StorageClasses
- StatefulSets cho stateful applications
- Cloud storage integration (EBS, GCS, Azure Disk)
- Shared storage (NFS, Ceph) cho distributed training

**35. Helm charts cho ML applications?**
- Packaging ML applications
- Templating và values.yaml
- Chart dependencies
- Release management

```yaml
# values.yaml
image:
  repository: mymodel
  tag: v1.0.0
  pullPolicy: IfNotPresent

replicaCount: 3

resources:
  requests:
    memory: 2Gi
    cpu: 1
  limits:
    memory: 4Gi
    cpu: 2

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70

model:
  path: /models/model.pkl
  version: v1

mlflow:
  trackingUri: http://mlflow-service:5000
```

**36. Namespaces và multi-tenancy cho ML teams?**
- Isolation giữa teams/projects
- Resource quotas per namespace
- RBAC (Role-Based Access Control)
- Network policies

**37. Monitoring ML models trên Kubernetes?**
- Prometheus cho metrics collection
- Grafana dashboards
- Custom metrics (prediction latency, throughput)
- Model-specific metrics (accuracy, drift)
- Logging với EFK stack

**38. Canary deployments và A/B testing cho models?**

```yaml
# Istio VirtualService for A/B testing
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-routing
spec:
  hosts:
  - model-service
  http:
  - match:
    - headers:
        x-user-group:
          exact: beta
    route:
    - destination:
        host: model-service
        subset: v2
  - route:
    - destination:
        host: model-service
        subset: v1
      weight: 90
    - destination:
        host: model-service
        subset: v2
      weight: 10
```

**39. Distributed training trên Kubernetes?**
- Kubeflow Training Operators (TFJob, PyTorchJob)
- Horovod với MPI Operator
- Parameter servers và workers
- Multi-node GPU training

**40. Service Mesh (Istio) benefits cho ML services?**
- Traffic management
- Security (mTLS)
- Observability
- Circuit breaking và retries
- Progressive delivery

---

## MLOps Tools & Practices

### Experiment Tracking & Model Registry

**41. MLflow components và use cases?**
- MLflow Tracking: experiments, parameters, metrics
- MLflow Projects: packaging ML code
- MLflow Models: deployment formats
- MLflow Registry: model lifecycle management

**42. Thiết kế model versioning strategy?**
- Semantic versioning
- Git tags for code
- Data versioning (DVC)
- Model registry stages (staging, production)
- Lineage tracking

**43. Weights & Biases vs MLflow vs Neptune?**
- Feature comparison
- Use cases
- Integration với training frameworks
- Collaboration features

### Data Management

**44. DVC (Data Version Control) là gì?**
- Git cho data và models
- Remote storage backends
- Pipeline management
- Reproducibility

**45. Data validation tools và strategies?**
- Great Expectations
- TensorFlow Data Validation (TFDV)
- Schema validation
- Data quality metrics
- Anomaly detection

**46. Feature stores và tại sao cần thiết?**
- Feast, Tecton, Hopsworks
- Centralized feature management
- Training-serving skew prevention
- Feature sharing across teams
- Online vs offline features

### Model Serving

**47. So sánh các model serving frameworks?**
- TensorFlow Serving
- TorchServe
- Triton Inference Server
- Seldon Core
- KServe (KFServing)
- BentoML

**48. Batch prediction vs Online prediction?**
- Use cases
- Infrastructure requirements
- Latency considerations
- Cost optimization

**49. Model optimization techniques cho production?**
- Quantization
- Pruning
- Knowledge distillation
- ONNX conversion
- TensorRT optimization

### Monitoring & Observability

**50. Model monitoring strategies?**
```python
# Example monitoring metrics
monitoring_metrics = {
    # Performance metrics
    "latency_p50": 50,  # ms
    "latency_p95": 100,
    "latency_p99": 200,
    "throughput": 1000,  # requests/sec

    # Model metrics
    "prediction_distribution": {...},
    "confidence_scores": {...},

    # Data quality
    "null_values_rate": 0.01,
    "feature_drift_score": 0.05,

    # Business metrics
    "prediction_accuracy": 0.95,
    "conversion_rate": 0.12,
}
```

**51. Data drift detection?**
- Statistical tests (KS test, Chi-square)
- Distribution comparisons
- Alerting mechanisms
- Tools: Evidently AI, NannyML, WhyLabs

**52. Model performance degradation detection?**
- Ground truth comparison
- Proxy metrics
- Concept drift vs data drift
- Automated retraining triggers

### Infrastructure as Code

**53. Terraform cho ML infrastructure?**
```hcl
# Example: Provision GPU cluster
resource "google_container_cluster" "ml_cluster" {
  name     = "ml-training-cluster"
  location = "us-central1"

  node_pool {
    name       = "gpu-pool"
    node_count = 2

    node_config {
      machine_type = "n1-standard-8"

      guest_accelerator {
        type  = "nvidia-tesla-v100"
        count = 1
      }

      oauth_scopes = [
        "https://www.googleapis.com/auth/cloud-platform"
      ]
    }
  }
}
```

**54. Ansible cho ML environment setup?**
- Configuration management
- Package installation
- Environment consistency
- GPU drivers và CUDA setup

### Workflow Orchestration

**55. So sánh Airflow, Kubeflow Pipelines, và Argo Workflows?**
- Architecture
- DAG definition
- Scheduling capabilities
- Kubernetes integration
- Use cases

**56. Thiết kế end-to-end ML pipeline với Kubeflow?**

```python
# Kubeflow pipeline example
from kfp import dsl
from kfp import components

@dsl.component
def preprocess_data(input_path: str, output_path: str):
    # preprocessing logic
    pass

@dsl.component
def train_model(data_path: str, model_output: str,
                epochs: int, learning_rate: float):
    # training logic
    pass

@dsl.component
def evaluate_model(model_path: str, test_data: str) -> float:
    # evaluation logic
    pass

@dsl.component
def deploy_model(model_path: str, accuracy: float,
                 threshold: float = 0.9):
    if accuracy >= threshold:
        # deploy to production
        pass

@dsl.pipeline(
    name='ML Training Pipeline',
    description='Complete ML pipeline'
)
def ml_pipeline(
    data_path: str,
    epochs: int = 10,
    learning_rate: float = 0.001
):
    preprocess_task = preprocess_data(
        input_path=data_path,
        output_path='/data/processed'
    )

    train_task = train_model(
        data_path=preprocess_task.outputs['output_path'],
        model_output='/models/trained',
        epochs=epochs,
        learning_rate=learning_rate
    )

    eval_task = evaluate_model(
        model_path=train_task.outputs['model_output'],
        test_data='/data/test'
    )

    deploy_task = deploy_model(
        model_path=train_task.outputs['model_output'],
        accuracy=eval_task.output
    )
```

**57. Apache Airflow DAG cho ML pipeline?**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'ml_pipeline',
    default_args=default_args,
    description='ML training and deployment pipeline',
    schedule_interval='@weekly',
    catchup=False
) as dag:

    extract_data = DockerOperator(
        task_id='extract_data',
        image='data-extractor:latest',
        api_version='auto',
        auto_remove=True,
        environment={
            'DB_HOST': '{{ var.value.db_host }}',
            'OUTPUT_PATH': '/data/raw'
        }
    )

    validate_data = DockerOperator(
        task_id='validate_data',
        image='data-validator:latest',
        api_version='auto',
        auto_remove=True
    )

    train_model = DockerOperator(
        task_id='train_model',
        image='model-trainer:latest',
        api_version='auto',
        auto_remove=True,
        device_requests=[
            {'driver': 'nvidia', 'count': 1, 'capabilities': [['gpu']]}
        ]
    )

    evaluate_model = PythonOperator(
        task_id='evaluate_model',
        python_callable=evaluate_model_func
    )

    deploy_model = DockerOperator(
        task_id='deploy_model',
        image='model-deployer:latest',
        api_version='auto',
        auto_remove=True
    )

    extract_data >> validate_data >> train_model >> evaluate_model >> deploy_model
```

### Security & Compliance

**58. Security best practices cho ML systems?**
- Model poisoning prevention
- Adversarial attack detection
- Data privacy (encryption at rest/transit)
- Access control (RBAC)
- Audit logging
- Secrets management (Vault, AWS Secrets Manager)

**59. GDPR compliance cho ML systems?**
- Right to explanation
- Data retention policies
- PII handling
- Model bias và fairness
- Consent management

**60. Model governance và audit trails?**
- Model cards
- Data lineage
- Experiment tracking
- Approval workflows
- Compliance reporting

---

## Câu hỏi tình huống

**61. Tình huống: Model accuracy đột ngột giảm trong production. Bạn xử lý như thế nào?**

Trả lời cần bao gồm:
1. Immediate actions:
   - Check monitoring dashboards
   - Compare current vs historical metrics
   - Check for infrastructure issues
   - Rollback to previous version if critical

2. Investigation:
   - Analyze input data distribution (data drift?)
   - Check for data quality issues
   - Review recent changes (code, config, infrastructure)
   - Compare predictions between versions
   - Analyze error patterns

3. Resolution:
   - Fix data pipeline issues if found
   - Retrain model with recent data if drift detected
   - Fix bugs if code issues found
   - Implement additional monitoring

4. Prevention:
   - Add drift detection alerts
   - Implement gradual rollouts
   - Improve testing coverage
   - Schedule regular retraining

**62. Tình huống: Triển khai ML service cần handle 10,000 requests/second với latency < 100ms**

Architecture suggestions:
- Load balancer (Nginx, AWS ALB)
- Kubernetes HPA cho auto-scaling
- Model optimization (quantization, TensorRT)
- Caching layer (Redis) cho frequent requests
- Batch prediction aggregation
- GPU inference optimization
- CDN cho edge deployment
- Monitoring với SLO/SLA tracking

**63. Tình huống: Team có nhiều data scientists, làm sao ensure reproducibility?**

Solutions:
- Containerization (Docker) cho environments
- Version control (Git) cho code
- Data versioning (DVC)
- Experiment tracking (MLflow)
- Pipeline orchestration (Kubeflow, Airflow)
- Shared compute resources (Kubernetes)
- Documentation standards
- Code review process
- CI/CD pipelines

**64. Tình huống: Training job cần 3 ngày trên single GPU. Optimize như thế nào?**

Approaches:
- Distributed training (multi-GPU, multi-node)
- Data parallelism (PyTorch DDP, Horovod)
- Model parallelism for large models
- Mixed precision training (FP16)
- Gradient accumulation
- Optimize data loading pipeline
- Use faster GPUs (V100 → A100)
- Hyperparameter optimization parallel
- Early stopping strategies
- Transfer learning từ pretrained models

**65. Tình huống: Deploy model mới nhưng không muốn risk production traffic**

Strategies:
- Blue-Green deployment
- Canary deployment (5% → 20% → 50% → 100%)
- Shadow mode (log predictions, không affect users)
- A/B testing với control group
- Feature flags
- Automated rollback based on metrics
- Gradual traffic shifting với Istio
- Multi-armed bandit approach

**66. Tình huống: Multiple models cần share features. Làm sao tránh duplication?**

Solution: Feature Store
- Centralized feature repository (Feast, Tecton)
- Online features cho real-time serving
- Offline features cho training
- Feature versioning
- Shared feature pipelines
- Documentation và discovery
- Consistency between training/serving

**67. Tình huống: Kubernetes cluster cost quá cao. Optimize như thế nào?**

Cost optimization:
- Right-sizing pods (requests/limits)
- Spot instances cho training jobs
- Cluster autoscaler
- Pod autoscaler (HPA, VPA)
- Bin packing optimization
- Remove unused resources
- Scheduled scaling (lower resources off-peak)
- Use cheaper regions/zones
- Reserved instances cho base load
- Monitor và alert on cost anomalies

**68. Tình huống: Data scientists muốn experiment nhanh nhưng infrastructure team lo security**

Balance approach:
- Self-service platforms (Kubeflow, SageMaker)
- Pre-approved base images
- Resource quotas per team
- Namespace isolation
- RBAC policies
- Automated security scanning
- Policy enforcement (OPA)
- Golden path templates
- Internal documentation
- Regular security training

**69. Tình huống: Model serving latency tăng đột ngột từ 50ms lên 300ms**

Debug steps:
1. Check infrastructure:
   - CPU/Memory utilization
   - Network latency
   - Pod health
   - Node capacity

2. Check application:
   - Request rate changes
   - Input data size changes
   - Model loading issues
   - Database query performance

3. Solutions based on findings:
   - Scale up pods if resource constrained
   - Add caching if repeated requests
   - Optimize model inference
   - Add request queuing
   - Implement rate limiting
   - Profile code for bottlenecks

**70. Tình huống: Cần retrain model mỗi ngày với fresh data automatically**

Implementation:
```python
# Airflow DAG cho daily retraining
from airflow import DAG
from datetime import datetime, timedelta

with DAG(
    'daily_model_retraining',
    default_args={
        'owner': 'ml-team',
        'retries': 2,
        'retry_delay': timedelta(minutes=5),
    },
    description='Daily model retraining pipeline',
    schedule_interval='0 2 * * *',  # 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False
) as dag:

    # Extract yesterday's data
    extract = DockerOperator(
        task_id='extract_data',
        image='data-extractor:latest',
        environment={
            'DATE': '{{ ds }}',  # Execution date
        }
    )

    # Validate data quality
    validate = DockerOperator(
        task_id='validate_data',
        image='data-validator:latest'
    )

    # Train new model
    train = DockerOperator(
        task_id='train_model',
        image='trainer:latest',
        device_requests=[
            {'driver': 'nvidia', 'count': 1, 'capabilities': [['gpu']]}
        ]
    )

    # Evaluate against production model
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=compare_models
    )

    # Deploy if better
    deploy = DockerOperator(
        task_id='deploy_if_better',
        image='deployer:latest',
        trigger_rule='all_success'
    )

    # Send notification
    notify = EmailOperator(
        task_id='notify_team',
        to='ml-team@company.com',
        subject='Daily retraining completed',
        html_content='Model retraining status: {{ task_instance.xcom_pull(task_ids="evaluate_model") }}'
    )

    extract >> validate >> train >> evaluate >> deploy >> notify
```

---

## Additional Topics

### 71. Model Explainability trong Production
- SHAP values computation
- LIME for local explanations
- Explainability APIs
- Performance vs interpretability trade-offs

### 72. Multi-model Serving
- Model ensemble strategies
- Routing logic
- Version management
- Resource allocation

### 73. Edge ML Deployment
- Model compression techniques
- TensorFlow Lite, ONNX Runtime
- Edge device constraints
- Offline inference

### 74. ML Platform Architecture
- Components: Training, Serving, Monitoring, Registry
- Self-service capabilities
- Scalability considerations
- Multi-tenancy support

### 75. Disaster Recovery cho ML Systems
- Model registry backups
- Data backups
- Infrastructure as Code
- Runbook documentation
- RTO/RPO requirements

---

## Tips cho người phỏng vấn

### Đánh giá candidates:

**Junior Level (0-2 years):**
- Hiểu basic Docker, Docker Compose
- Familiar với CI/CD concepts
- Basic Kubernetes knowledge (Pods, Deployments, Services)
- Biết một vài ML tools (MLflow hoặc similar)

**Mid Level (2-4 years):**
- Hands-on experience với containerization
- Thiết kế và implement CI/CD pipelines
- Deploy và manage apps trên Kubernetes
- Experience với experiment tracking và model registry
- Understanding của ML lifecycle

**Senior Level (4+ years):**
- Architect end-to-end MLOps platforms
- Advanced Kubernetes (operators, custom resources)
- Design distributed training systems
- Implement comprehensive monitoring
- Cost optimization và security
- Mentoring và best practices advocacy

### Red Flags:
- Không có hands-on experience, chỉ theoretical
- Không hiểu reproducibility importance
- Ignore monitoring và observability
- Không consider security
- Không có version control discipline

### Green Flags:
- Production experience
- Understanding của trade-offs
- Automation mindset
- Monitoring-first approach
- Security awareness
- Cost consciousness
- Good documentation habits

---

**Tổng kết:** Bộ câu hỏi này cover từ basics đến advanced topics trong MLOps, bao gồm cả technical knowledge và practical experience. Candidates nên demonstrate không chỉ theoretical understanding mà còn hands-on experience và ability to make trade-offs based on requirements.
