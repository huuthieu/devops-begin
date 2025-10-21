# Câu Hỏi Tình Huống

[← Quay lại](./README.md) | [← MLOps Tools](./05-mlops-tools.md) | [Additional Topics →](./07-additional-topics.md)

## Production Issues & Troubleshooting

### 61. Tình huống: Model accuracy đột ngột giảm trong production. Bạn xử lý như thế nào?

**Approach:**

**1. Immediate Actions:**
- Check monitoring dashboards (Grafana, Datadog)
- Compare current vs historical metrics
- Check for infrastructure issues (CPU, memory, errors)
- Rollback to previous version if critical impact

**2. Investigation:**
- **Data drift analysis:**
  ```python
  from evidently.metrics import DataDriftPreset
  report = Report(metrics=[DataDriftPreset()])
  report.run(reference_data=train_df, current_data=prod_df)
  ```
- Check for data quality issues (nulls, outliers)
- Review recent changes (code, config, infrastructure)
- Compare predictions between versions
- Analyze error patterns and failure cases

**3. Resolution:**
- Fix data pipeline issues if found
- Retrain model with recent data if drift detected
- Fix bugs if code issues found
- Implement additional monitoring

**4. Prevention:**
- Add drift detection alerts
- Implement gradual rollouts (canary deployments)
- Improve testing coverage
- Schedule regular retraining
- Add data quality checks in pipeline

---

### 62. Tình huống: Triển khai ML service cần handle 10,000 requests/second với latency < 100ms

**Architecture Design:**

**1. Infrastructure:**
```yaml
# Kubernetes HPA for auto-scaling
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-hpa
spec:
  scaleTargetRef:
    kind: Deployment
    name: model-api
  minReplicas: 10
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        averageUtilization: 70
```

**2. Optimization Strategies:**
- **Load balancer:** Nginx or AWS ALB
- **Auto-scaling:** Kubernetes HPA
- **Model optimization:**
  - Quantization (FP32 → INT8)
  - TensorRT for GPU inference
  - ONNX Runtime
- **Caching:** Redis for frequent requests
- **Batch aggregation:** Group requests (if latency allows)
- **GPU optimization:** Use Triton Inference Server
- **CDN:** Edge deployment for global users

**3. Monitoring:**
```python
# SLO/SLA tracking
SLO = {
    "latency_p99": 100,  # ms
    "availability": 99.9,  # %
    "throughput": 10000,  # req/s
}
```

---

### 63. Tình huống: Team có nhiều data scientists, làm sao ensure reproducibility?

**Solutions:**

**1. Environment Management:**
```dockerfile
# Docker for consistent environments
FROM python:3.9-slim
COPY requirements.txt .
RUN pip install -r requirements.txt
# Ensures everyone has same dependencies
```

**2. Version Control:**
- **Code:** Git with clear branching strategy
- **Data:** DVC
  ```bash
  dvc add data/train.csv
  dvc push
  ```
- **Models:** MLflow Registry
  ```python
  mlflow.log_model(model, "model")
  mlflow.register_model("runs:/123/model", "MyModel")
  ```

**3. Experiment Tracking:**
```python
import mlflow

with mlflow.start_run():
    mlflow.log_params({"lr": 0.01, "epochs": 10})
    mlflow.log_metrics({"accuracy": 0.95})
    mlflow.log_artifact("model.pkl")
```

**4. Pipeline Orchestration:**
- Use Kubeflow or Airflow
- Declarative pipelines
- Shared compute resources (Kubernetes)

**5. Best Practices:**
- Documentation standards
- Code review process
- CI/CD pipelines
- Shared notebooks repo
- Regular sync meetings

---

## Performance & Optimization

### 64. Tình huống: Training job cần 3 ngày trên single GPU. Optimize như thế nào?

**Optimization Approaches:**

**1. Distributed Training:**
```python
# PyTorch DDP example
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

dist.init_process_group(backend='nccl')
model = DistributedDataParallel(model)
```

**2. Data Parallelism:**
- PyTorch DistributedDataParallel
- Horovod
- Scale to 4-8 GPUs → 4-8x speedup

**3. Model Parallelism:**
- For models too large for single GPU
- Split model across GPUs
- Pipeline parallelism

**4. Mixed Precision Training:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for data, target in dataloader:
    with autocast():
        output = model(data)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```
- 2-3x speedup with minimal accuracy loss

**5. Other Optimizations:**
- Gradient accumulation (simulate larger batch)
- Optimize data loading pipeline (multiple workers)
- Use faster GPUs (V100 → A100)
- Hyperparameter optimization in parallel
- Early stopping strategies
- Transfer learning from pretrained models

**Expected Results:**
- 3 days → 6-12 hours with optimizations

---

### 65. Tình huống: Deploy model mới nhưng không muốn risk production traffic

**Deployment Strategies:**

**1. Blue-Green Deployment:**
```yaml
# Keep both versions running
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-api
    version: blue  # Switch to green when ready
```

**2. Canary Deployment:**
```yaml
# Istio VirtualService - gradual rollout
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: model-canary
spec:
  http:
  - route:
    - destination:
        host: model-v1
      weight: 95  # Old version
    - destination:
        host: model-v2
      weight: 5   # New version - increase gradually
```

**Rollout plan:**
- Day 1: 5% traffic
- Day 2: 20% traffic (if metrics good)
- Day 3: 50% traffic
- Day 4: 100% traffic

**3. Shadow Mode:**
```python
# Log predictions without affecting users
async def predict(request):
    # Production model
    prod_prediction = prod_model.predict(request)

    # New model (shadow)
    asyncio.create_task(
        new_model.predict_and_log(request)
    )

    return prod_prediction
```

**4. Feature Flags:**
```python
if feature_flags.is_enabled("new_model", user_id):
    return new_model.predict(data)
else:
    return old_model.predict(data)
```

**5. Automated Rollback:**
```python
if metrics["accuracy"] < 0.90 or metrics["latency_p95"] > 200:
    rollback_to_previous_version()
    alert_team()
```

---

## Cost & Resource Optimization

### 66. Tình huống: Multiple models cần share features. Làm sao tránh duplication?

**Solution: Feature Store**

**Architecture:**
```python
from feast import FeatureStore

# Define features once
store = FeatureStore(repo_path=".")

# Training: offline features
training_df = store.get_historical_features(
    entity_df=entity_df,
    features=[
        "user_features:age",
        "user_features:country",
        "transaction_features:amount_7d_avg",
    ],
).to_df()

# Serving: online features
features = store.get_online_features(
    features=[
        "user_features:age",
        "user_features:country",
    ],
    entity_rows=[{"user_id": "123"}],
).to_dict()
```

**Benefits:**
- **Consistency:** Same features for training and serving
- **Reusability:** Multiple models share features
- **Versioning:** Track feature changes
- **Documentation:** Centralized feature definitions
- **Efficiency:** Computed once, used by many

**Popular Solutions:**
- Feast (open source)
- Tecton
- AWS SageMaker Feature Store
- GCP Vertex AI Feature Store

---

### 67. Tình huống: Kubernetes cluster cost quá cao. Optimize như thế nào?

**Cost Optimization Strategies:**

**1. Right-sizing Pods:**
```yaml
# Before: over-provisioned
resources:
  requests:
    memory: "16Gi"  # Actually using 4Gi
    cpu: "8"        # Actually using 2
  limits:
    memory: "32Gi"
    cpu: "16"

# After: right-sized
resources:
  requests:
    memory: "4Gi"
    cpu: "2"
  limits:
    memory: "8Gi"
    cpu: "4"
```

**2. Spot Instances for Training:**
```yaml
# Use spot instances for interruptible workloads
nodeSelector:
  cloud.google.com/gke-preemptible: "true"
tolerations:
- key: "cloud.google.com/gke-preemptible"
  operator: "Equal"
  value: "true"
  effect: "NoSchedule"
```
- 60-80% cost reduction for training jobs

**3. Auto-scaling:**
- Cluster Autoscaler (add/remove nodes)
- Horizontal Pod Autoscaler (scale pods)
- Vertical Pod Autoscaler (right-size requests)

**4. Scheduled Scaling:**
```python
# Scale down dev/staging environments off-hours
# 9 AM - 6 PM: 5 replicas
# 6 PM - 9 AM: 1 replica
# Weekends: 1 replica
```

**5. Resource Cleanup:**
- Remove unused PersistentVolumes
- Clean up old images
- Delete failed jobs/pods
- Remove unused namespaces

**6. Cost Monitoring:**
```yaml
# Add cost labels
metadata:
  labels:
    team: ml-team
    env: production
    cost-center: research
```

**Expected Savings:**
- 40-60% cost reduction possible

---

## Security & Team Collaboration

### 68. Tình huống: Data scientists muốn experiment nhanh nhưng infrastructure team lo security

**Balanced Approach:**

**1. Self-Service Platform:**
```yaml
# Kubeflow Notebooks - managed Jupyter
apiVersion: kubeflow.org/v1
kind: Notebook
metadata:
  name: user-notebook
spec:
  template:
    spec:
      containers:
      - image: approved-jupyter-image:v1
        resources:
          limits:
            nvidia.com/gpu: "1"
```

**2. Pre-approved Base Images:**
```dockerfile
# Approved images with security scanning
FROM company/ml-base:python3.9-gpu
# Pre-installed: tensorflow, pytorch, scikit-learn
# Security: scanned, non-root user, minimal packages
```

**3. Resource Quotas:**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-experiments
spec:
  hard:
    requests.cpu: "50"
    requests.memory: 100Gi
    requests.nvidia.com/gpu: "4"
    persistentvolumeclaims: "10"
```

**4. RBAC Policies:**
```yaml
# Users can create pods, but not modify network policies
kind: Role
metadata:
  namespace: ml-experiments
rules:
- apiGroups: [""]
  resources: ["pods", "configmaps"]
  verbs: ["create", "get", "list"]
- apiGroups: [""]
  resources: ["networkpolicies"]
  verbs: ["get", "list"]  # Read-only
```

**5. Golden Path Templates:**
```python
# cookiecutter template for new projects
project/
├── Dockerfile (pre-approved)
├── requirements.txt
├── src/
├── tests/
└── k8s/
    ├── deployment.yaml (templated)
    └── service.yaml
```

**6. Automated Security:**
- Container scanning in CI/CD
- Policy enforcement (OPA/Gatekeeper)
- Secret scanning (git-secrets)

---

### 69. Tình huống: Model serving latency tăng đột ngột từ 50ms lên 300ms

**Debugging Process:**

**1. Check Infrastructure:**
```bash
# Pod resource utilization
kubectl top pods -n production

# Pod status
kubectl get pods -n production

# Node capacity
kubectl describe node <node-name>

# Network latency
ping <service-endpoint>
```

**2. Check Application Logs:**
```bash
# Recent errors
kubectl logs deployment/model-api --tail=100 | grep ERROR

# Slow requests
kubectl logs deployment/model-api | grep "latency > 200ms"
```

**3. Common Causes & Solutions:**

| Cause | Detection | Solution |
|-------|-----------|----------|
| CPU throttling | `kubectl top pods` | Increase CPU limits |
| Memory pressure | OOM events in logs | Increase memory |
| Cold start | First request slow | Keep warm instances |
| Large batch size | Correlation with request size | Reduce batch size |
| Model loading | Startup latency | Cache model in memory |
| Database slow | DB query logs | Add indexes, caching |
| Network issues | Network monitoring | Check service mesh |

**4. Profiling:**
```python
import cProfile

def profile_prediction():
    profiler = cProfile.Profile()
    profiler.enable()

    result = model.predict(data)

    profiler.disable()
    profiler.print_stats(sort='cumulative')

# Find bottleneck: preprocessing, inference, or postprocessing?
```

**5. Solutions Based on Findings:**
- Scale up pods if resource constrained
- Add Redis caching for repeated requests
- Optimize model inference (quantization, TensorRT)
- Add request queuing with rate limiting
- Profile code for bottlenecks

---

### 70. Tình huống: Cần retrain model mỗi ngày với fresh data automatically

**Implementation:**

**Airflow DAG:**
```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.docker.operators.docker import DockerOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'ml-team',
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': True,
    'email': ['ml-team@company.com'],
}

with DAG(
    'daily_model_retraining',
    default_args=default_args,
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
            'OUTPUT_PATH': '/data/raw/{{ ds }}'
        }
    )

    # Validate data quality
    validate = PythonOperator(
        task_id='validate_data',
        python_callable=validate_data_quality,
        op_kwargs={'date': '{{ ds }}'}
    )

    # Train new model
    train = DockerOperator(
        task_id='train_model',
        image='trainer:latest',
        environment={
            'DATA_PATH': '/data/processed/{{ ds }}',
            'MODEL_OUTPUT': '/models/{{ ds }}'
        },
        device_requests=[
            {'driver': 'nvidia', 'count': 1, 'capabilities': [['gpu']]}
        ]
    )

    # Evaluate against production model
    evaluate = PythonOperator(
        task_id='evaluate_model',
        python_callable=compare_with_production,
        op_kwargs={
            'new_model_path': '/models/{{ ds }}',
            'prod_model_path': '/models/production'
        }
    )

    # Deploy if better
    deploy = PythonOperator(
        task_id='deploy_if_better',
        python_callable=conditional_deploy,
        trigger_rule='all_success'
    )

    # Send notification
    notify = PythonOperator(
        task_id='notify_team',
        python_callable=send_slack_notification
    )

    extract >> validate >> train >> evaluate >> deploy >> notify
```

**Key Features:**
- Automated daily execution
- Data validation gates
- Model comparison before deployment
- Retry logic for failures
- Notifications to team
- GPU support for training

---

[← MLOps Tools](./05-mlops-tools.md) | [Additional Topics →](./07-additional-topics.md)
