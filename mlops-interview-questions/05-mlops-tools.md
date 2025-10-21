# MLOps Tools & Practices

[← Quay lại](./README.md) | [← Kubernetes](./04-kubernetes.md) | [Scenarios →](./06-scenarios.md)

## Experiment Tracking & Model Registry

### 41. MLflow components và use cases?

**Components:**
- **MLflow Tracking:** log experiments, parameters, metrics, artifacts
- **MLflow Projects:** packaging ML code với dependencies
- **MLflow Models:** deployment formats (Python function, REST API, etc.)
- **MLflow Registry:** model lifecycle management (staging, production)

**Use cases:**
- Experiment comparison
- Model versioning
- Deployment automation
- Collaboration

### 42. Thiết kế model versioning strategy?

**Strategy components:**
- **Code versioning:** Git tags, semantic versioning
- **Data versioning:** DVC, git-lfs
- **Model versioning:** Model registry (MLflow, Neptune)
- **Metadata:** training date, metrics, hyperparameters
- **Lineage tracking:** data → code → model mapping

**Example:**
```
model-v1.2.3
├── code: git commit sha123abc
├── data: dvc version abc123
├── metrics: accuracy=0.95, f1=0.93
└── stage: production
```

### 43. Weights & Biases vs MLflow vs Neptune?

**Comparison:**

| Feature | W&B | MLflow | Neptune |
|---------|-----|--------|---------|
| Experiment tracking | ✅ Excellent | ✅ Good | ✅ Excellent |
| Visualization | ✅ Rich | ⚠️ Basic | ✅ Good |
| Collaboration | ✅ Strong | ⚠️ Limited | ✅ Strong |
| Model registry | ✅ Yes | ✅ Yes | ✅ Yes |
| Self-hosted | ⚠️ Limited | ✅ Full | ⚠️ Limited |
| Cost | $$ Paid tiers | Free OSS | $$ Paid tiers |

---

## Data Management

### 44. DVC (Data Version Control) là gì?

**Key features:**
- Git for data và models
- Remote storage backends (S3, GCS, Azure, NFS)
- Pipeline management
- Reproducibility

**Basic workflow:**
```bash
# Initialize DVC
dvc init

# Add data
dvc add data/train.csv

# Configure remote storage
dvc remote add -d storage s3://mybucket/dvc-store

# Push data
dvc push

# Pull data
dvc pull
```

### 45. Data validation tools và strategies?

**Tools:**
- **Great Expectations:** comprehensive data validation
- **TensorFlow Data Validation (TFDV):** schema inference, anomaly detection
- **Pandera:** lightweight DataFrame validation
- **Deepchecks:** ML-specific validation

**Validation strategies:**
1. Schema validation (data types, columns)
2. Range checks (min, max values)
3. Distribution checks
4. Null/missing values
5. Uniqueness constraints
6. Relationships between features

### 46. Feature stores và tại sao cần thiết?

**Popular feature stores:**
- Feast (open source)
- Tecton (commercial)
- Hopsworks
- AWS SageMaker Feature Store
- GCP Vertex AI Feature Store

**Benefits:**
- Centralized feature management
- Training-serving skew prevention
- Feature sharing across teams
- Feature versioning
- Online (real-time) vs offline (batch) features
- Feature lineage

---

## Model Serving

### 47. So sánh các model serving frameworks?

**Framework comparison:**

| Framework | Best For | Protocols | Features |
|-----------|----------|-----------|----------|
| TensorFlow Serving | TF models | REST, gRPC | Model versioning, batching |
| TorchServe | PyTorch | REST, gRPC | Multi-model, metrics |
| Triton | Multi-framework | REST, gRPC | GPU optimization, ensembles |
| Seldon Core | K8s native | REST, gRPC | A/B testing, explainability |
| KServe | Serverless | REST, gRPC | Auto-scaling, canary |
| BentoML | Custom logic | REST | Easy packaging, deployment |

### 48. Batch prediction vs Online prediction?

**Batch prediction:**
- **Use cases:** nightly scoring, bulk processing
- **Latency:** hours to days
- **Infrastructure:** Spark, Beam, scheduled jobs
- **Cost:** lower per prediction
- **Example:** customer churn prediction

**Online prediction:**
- **Use cases:** real-time decisions
- **Latency:** milliseconds to seconds
- **Infrastructure:** REST API, model servers
- **Cost:** higher per prediction
- **Example:** fraud detection, recommendations

### 49. Model optimization techniques cho production?

**Techniques:**

1. **Quantization:**
   - FP32 → FP16 (half precision)
   - INT8 quantization
   - Trade-off: size/speed vs accuracy

2. **Pruning:**
   - Remove less important weights
   - Structured vs unstructured
   - Can reduce size by 90%+

3. **Knowledge Distillation:**
   - Train smaller model from larger model
   - Maintains most performance

4. **Model Compilation:**
   - ONNX conversion (cross-framework)
   - TensorRT (NVIDIA GPUs)
   - OpenVINO (Intel)

5. **Architecture Search:**
   - EfficientNet, MobileNet for mobile/edge

---

## Monitoring & Observability

### 50. Model monitoring strategies?

**Metrics to monitor:**
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

**Monitoring layers:**
1. Infrastructure (CPU, memory, GPU)
2. Application (latency, errors)
3. Model performance (accuracy, drift)
4. Business impact (conversion, revenue)

### 51. Data drift detection?

**Detection methods:**
- **Statistical tests:**
  - Kolmogorov-Smirnov test
  - Chi-square test
  - Population Stability Index (PSI)
- **Distribution comparisons:**
  - KL divergence
  - Wasserstein distance

**Tools:**
- Evidently AI
- NannyML
- WhyLabs
- Fiddler

**Implementation:**
```python
from evidently.metrics import DataDriftPreset
from evidently.report import Report

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=train_df, current_data=prod_df)
```

### 52. Model performance degradation detection?

**Approaches:**

1. **Ground truth comparison:**
   - Collect labels for predictions
   - Compare against actuals
   - Delayed feedback challenge

2. **Proxy metrics:**
   - User engagement
   - Click-through rates
   - Business KPIs

3. **Drift types:**
   - **Data drift:** input distribution changes
   - **Concept drift:** relationship X→Y changes
   - **Prediction drift:** output distribution changes

**Automated responses:**
- Alert ML team
- Trigger retraining
- Rollback to previous version
- Switch to challenger model

---

## Infrastructure as Code

### 53. Terraform cho ML infrastructure?

**Example: Provision GPU cluster**
```hcl
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

resource "google_storage_bucket" "model_artifacts" {
  name     = "ml-model-artifacts"
  location = "US"

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
}
```

### 54. Ansible cho ML environment setup?

**Use cases:**
- Configuration management
- Package installation (Python, CUDA, drivers)
- Environment consistency across nodes
- GPU drivers và CUDA setup
- Multi-machine setup

**Example playbook:**
```yaml
---
- name: Setup ML Environment
  hosts: ml_nodes
  tasks:
    - name: Install NVIDIA drivers
      apt:
        name: nvidia-driver-515
        state: present

    - name: Install CUDA
      shell: |
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
        mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
        apt-get install cuda

    - name: Install Python dependencies
      pip:
        requirements: /home/mluser/requirements.txt
```

---

## Workflow Orchestration

### 55. So sánh Airflow, Kubeflow Pipelines, và Argo Workflows?

**Comparison:**

| Feature | Airflow | Kubeflow | Argo |
|---------|---------|----------|------|
| **Focus** | General workflow | ML pipelines | K8s workflows |
| **DAG Definition** | Python | Python SDK | YAML |
| **Scheduling** | Built-in | External | External |
| **UI** | Good | Good | Basic |
| **K8s Native** | No | Yes | Yes |
| **Learning Curve** | Medium | Medium | Low |

**Use cases:**
- **Airflow:** ETL, data pipelines, scheduled ML retraining
- **Kubeflow:** End-to-end ML pipelines, hyperparameter tuning
- **Argo:** CI/CD, event-driven workflows

### 56. Thiết kế end-to-end ML pipeline với Kubeflow?

**Pipeline example:**
```python
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

### 57. Apache Airflow DAG cho ML pipeline?

**Complete DAG example:**
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

---

## Security & Compliance

### 58. Security best practices cho ML systems?

**Key practices:**

1. **Model Security:**
   - Model poisoning prevention
   - Adversarial attack detection
   - Input validation

2. **Data Privacy:**
   - Encryption at rest
   - Encryption in transit (TLS)
   - PII anonymization
   - Differential privacy

3. **Access Control:**
   - RBAC for models and data
   - Principle of least privilege
   - Audit logging

4. **Secrets Management:**
   - Vault, AWS Secrets Manager
   - Never hard-code credentials
   - Rotate credentials regularly

5. **Dependency Management:**
   - Scan for vulnerabilities
   - Keep dependencies updated
   - Use private package repositories

### 59. GDPR compliance cho ML systems?

**Requirements:**

1. **Right to Explanation:**
   - Model interpretability
   - Decision explanations
   - SHAP, LIME integration

2. **Data Retention:**
   - Automatic data deletion
   - Retention policies
   - Right to be forgotten

3. **Privacy by Design:**
   - Data minimization
   - Pseudonymization
   - Encryption

4. **Consent Management:**
   - Track user consent
   - Purpose limitation
   - Opt-out mechanisms

5. **Bias & Fairness:**
   - Fairness metrics
   - Regular audits
   - Demographic parity

### 60. Model governance và audit trails?

**Components:**

1. **Model Cards:**
   - Model description
   - Intended use
   - Training data
   - Performance metrics
   - Limitations

2. **Data Lineage:**
   - Track data sources
   - Transformation history
   - Version control

3. **Experiment Tracking:**
   - All training runs
   - Hyperparameters
   - Metrics

4. **Approval Workflows:**
   - Model review process
   - Stakeholder sign-off
   - Compliance checks

5. **Audit Logs:**
   - Who deployed what when
   - Model predictions (sampling)
   - Configuration changes

---

[← Kubernetes](./04-kubernetes.md) | [Scenarios →](./06-scenarios.md)
