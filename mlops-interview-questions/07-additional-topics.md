# Additional Topics

[← Quay lại](./README.md) | [← Scenarios](./06-scenarios.md) | [Interviewer Tips →](./08-interviewer-tips.md)

## Advanced MLOps Topics

### 71. Model Explainability trong Production

**Why Important:**
- Regulatory compliance (GDPR, financial services)
- Trust và adoption
- Debug model decisions
- Bias detection

**Techniques:**

**1. SHAP (SHapley Additive exPlanations):**
```python
import shap

# Create explainer
explainer = shap.TreeExplainer(model)

# Get SHAP values
shap_values = explainer.shap_values(X)

# Visualize
shap.summary_plot(shap_values, X)
```

**2. LIME (Local Interpretable Model-agnostic Explanations):**
```python
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=['0', '1']
)

explanation = explainer.explain_instance(
    X_test[0],
    model.predict_proba
)
```

**3. Production Implementation:**
```python
# API endpoint for explanations
@app.post("/predict-with-explanation")
async def predict_explain(data: InputData):
    prediction = model.predict(data)

    # Compute explanation
    explanation = explainer.explain_instance(data)

    return {
        "prediction": prediction,
        "explanation": {
            "top_features": explanation.top_features,
            "shap_values": explanation.shap_values
        }
    }
```

**Trade-offs:**
- Performance vs interpretability
- Latency impact (SHAP can be slow)
- Consider pre-computing for common cases
- Use simpler explainers for real-time (feature importance)

---

### 72. Multi-model Serving

**Use Cases:**
- Model ensembles
- A/B testing different models
- Specialized models per segment
- Shadow deployments

**Architecture Patterns:**

**1. Model Ensemble:**
```python
class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        # Voting or averaging
        return np.mean(predictions, axis=0)
```

**2. Router-based Selection:**
```python
class ModelRouter:
    def __init__(self):
        self.models = {
            'premium_users': premium_model,
            'standard_users': standard_model,
            'new_users': simple_model
        }

    def predict(self, X, user_segment):
        model = self.models[user_segment]
        return model.predict(X)
```

**3. Kubernetes Multi-model:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-router
spec:
  selector:
    app: model-router
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v1
spec:
  replicas: 3
  template:
    metadata:
      labels:
        version: v1
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-v2
spec:
  replicas: 3
  template:
    metadata:
      labels:
        version: v2
```

**Resource Allocation:**
- Shared resources vs isolated
- GPU sharing for multiple models
- Memory considerations
- Auto-scaling per model

---

### 73. Edge ML Deployment

**Challenges:**
- Limited compute resources
- No network connectivity (offline)
- Power constraints
- Model size restrictions

**Model Compression:**

**1. Quantization:**
```python
import tensorflow as tf

# Post-training quantization
converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Reduces model size by 4x (FP32 → INT8)
```

**2. Pruning:**
```python
import tensorflow_model_optimization as tfmot

# Prune model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=0.5,
        begin_step=0,
        end_step=1000
    )
}

model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
```

**3. Knowledge Distillation:**
```python
# Train small model from large model
class Distiller(tf.keras.Model):
    def train_step(self, data):
        x, y = data

        # Teacher predictions
        teacher_predictions = teacher_model(x, training=False)

        with tf.GradientTape() as tape:
            # Student predictions
            student_predictions = self(x, training=True)

            # Distillation loss
            loss = distillation_loss(
                student_predictions,
                teacher_predictions,
                y
            )

        # Update student
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}
```

**Deployment:**
```python
# TensorFlow Lite on mobile
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
```

**Frameworks:**
- TensorFlow Lite (mobile, embedded)
- ONNX Runtime (cross-platform)
- Core ML (iOS)
- TensorRT (NVIDIA edge devices)

---

### 74. ML Platform Architecture

**Components of End-to-End ML Platform:**

```
┌─────────────────────────────────────────────────────┐
│              ML Platform Architecture                │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Data       │  │  Feature     │  │  Experiment│ │
│  │  Management  │→ │   Store      │→ │  Tracking  │ │
│  │  (DVC, S3)   │  │  (Feast)     │  │  (MLflow)  │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│         ↓                                      ↓      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │   Training   │  │    Model     │  │   Model    │ │
│  │ Orchestration│→ │   Registry   │→ │  Serving   │ │
│  │ (Kubeflow)   │  │  (MLflow)    │  │ (Triton)   │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│                                             ↓         │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │  Monitoring  │← │   Serving    │← │    API     │ │
│  │  & Alerting  │  │Infrastructure│  │  Gateway   │ │
│  │ (Prometheus) │  │ (Kubernetes) │  │            │ │
│  └──────────────┘  └──────────────┘  └────────────┘ │
│                                                       │
└─────────────────────────────────────────────────────┘
```

**Key Features:**

**1. Self-Service Capabilities:**
- Jupyter notebook environments
- One-click training jobs
- Automated deployment pipelines
- Template projects

**2. Scalability:**
- Auto-scaling infrastructure
- Distributed training support
- Multi-tenancy
- Resource quotas

**3. Governance:**
- Model registry
- Experiment tracking
- Audit trails
- Access control

**4. Integration:**
- CI/CD pipelines
- Data sources
- Monitoring tools
- Cloud services

**Example: Platform API:**
```python
from ml_platform import Platform

platform = Platform()

# Create training job
job = platform.create_training_job(
    name="my-model-v1",
    code_uri="s3://bucket/code",
    data_uri="s3://bucket/data",
    instance_type="ml.p3.8xlarge",
    hyperparameters={
        "epochs": 10,
        "lr": 0.001
    }
)

# Deploy to production
endpoint = platform.deploy_model(
    model_uri=job.best_model_uri,
    instance_type="ml.c5.2xlarge",
    replicas=3,
    canary_percent=10
)
```

---

### 75. Disaster Recovery cho ML Systems

**RTO/RPO Requirements:**
- **RTO (Recovery Time Objective):** Maximum acceptable downtime
- **RPO (Recovery Point Objective):** Maximum acceptable data loss

**Strategy:**

**1. Model Registry Backups:**
```python
# Automated model backup
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Backup all production models
prod_models = client.search_model_versions("tags.stage='Production'")

for model in prod_models:
    # Backup to S3
    client.download_artifacts(
        model.run_id,
        model.source,
        dst_path=f"s3://backup/models/{model.name}/{model.version}"
    )
```

**2. Data Backups:**
```yaml
# Automated backup CronJob
apiVersion: batch/v1
kind: CronJob
metadata:
  name: data-backup
spec:
  schedule: "0 2 * * *"  # Daily at 2 AM
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: backup-tool:latest
            command:
            - /bin/sh
            - -c
            - |
              aws s3 sync /data s3://backup-bucket/data/$(date +%Y-%m-%d)
              aws s3 sync /models s3://backup-bucket/models/$(date +%Y-%m-%d)
```

**3. Infrastructure as Code:**
```hcl
# Terraform for infrastructure
# Can recreate entire infrastructure from code

module "ml_infrastructure" {
  source = "./modules/ml-infrastructure"

  cluster_name = "ml-production"
  node_count   = 5
  gpu_nodes    = 2

  # All configuration in version control
}
```

**4. Runbook Documentation:**
```markdown
# Disaster Recovery Runbook

## Scenario 1: Model Serving Outage

1. Check health endpoints
2. Review logs: `kubectl logs deployment/model-api`
3. Rollback if needed: `kubectl rollout undo deployment/model-api`
4. If total failure, restore from backup:
   ```bash
   kubectl apply -f backup/deployment.yaml
   ```

## Scenario 2: Data Corruption

1. Stop all writes to database
2. Identify last good backup
3. Restore from backup:
   ```bash
   aws s3 cp s3://backup-bucket/data/2024-01-15 /data --recursive
   ```
4. Verify data integrity
5. Resume operations

## Scenario 3: Complete Infrastructure Loss

1. Initialize Terraform
2. Apply infrastructure:
   ```bash
   terraform apply
   ```
3. Restore data from backups
4. Deploy models from model registry
5. Update DNS
```

**5. Testing:**
- Regular DR drills (quarterly)
- Automated restore testing
- Chaos engineering (deliberately break things)
- Measure actual RTO/RPO

**6. Multi-Region Setup:**
```yaml
# Active-passive setup
regions:
  primary:
    region: us-east-1
    status: active
    models: [v1, v2, v3]

  secondary:
    region: us-west-2
    status: standby
    models: [v1, v2, v3]  # Replicated
    failover: automatic
```

---

[← Scenarios](./06-scenarios.md) | [Interviewer Tips →](./08-interviewer-tips.md)
