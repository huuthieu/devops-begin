# Kubernetes

[← Quay lại](./README.md) | [← CI/CD](./03-cicd.md) | [MLOps Tools →](./05-mlops-tools.md)

## Câu hỏi cơ bản

### 26. Kubernetes là gì và tại sao dùng cho MLOps?

**Key points:**
- Container orchestration platform
- Auto-scaling, self-healing, load balancing
- Resource management cho training và serving
- Multi-environment support (dev, staging, prod)
- Declarative configuration

### 27. Các components chính của Kubernetes?

**Control Plane:**
- API Server: central management
- Scheduler: pod placement
- Controller Manager: desired state
- etcd: distributed key-value store

**Worker Nodes:**
- Kubelet: node agent
- Kube-proxy: network proxy
- Container Runtime: Docker, containerd

**Objects:**
- Pods, Services, Deployments, ReplicaSets

### 28. Pod là gì? Deployment là gì?

**Pod:**
- Smallest deployable unit
- 1+ containers sharing network/storage
- Ephemeral nature

**Deployment:**
- Manages ReplicaSets
- Declarative updates
- Scaling và rolling updates
- Rollback capabilities

### 29. Services trong Kubernetes và các types?

**Service types:**
- **ClusterIP:** internal communication only
- **NodePort:** external access via node port
- **LoadBalancer:** cloud load balancer
- **ExternalName:** DNS CNAME mapping

**Use cases:**
- ClusterIP: inter-service communication
- LoadBalancer: production external access

### 30. ConfigMaps và Secrets?

**ConfigMaps:**
- Non-sensitive configuration
- Key-value pairs
- Can be mounted as files or env vars

**Secrets:**
- Sensitive data (passwords, tokens)
- Base64 encoded
- Better access control

---

## Câu hỏi nâng cao

### 31. Deploy một ML model trên Kubernetes - các resources cần thiết?

**Complete deployment:**
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

### 32. Resource management cho ML workloads?

**Key concepts:**
- **Requests:** guaranteed resources
- **Limits:** maximum resources
- GPU resources: `nvidia.com/gpu: 1`
- Memory-intensive training jobs
- Resource quotas và limit ranges
- Node affinity cho GPU nodes

**Example:**
```yaml
resources:
  requests:
    memory: "16Gi"
    cpu: "4"
    nvidia.com/gpu: "1"
  limits:
    memory: "32Gi"
    cpu: "8"
    nvidia.com/gpu: "1"
```

### 33. Training jobs trên Kubernetes - Job vs CronJob?

**Job example:**
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
```

**CronJob for scheduled retraining:**
```yaml
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

### 34. Persistent storage cho ML trong Kubernetes?

**Storage options:**
- **PersistentVolume (PV):** cluster-level storage
- **PersistentVolumeClaim (PVC):** user request for storage
- **StorageClasses:** dynamic provisioning
- **StatefulSets:** for stateful applications

**Cloud integrations:**
- AWS EBS, EFS
- GCP Persistent Disk, Filestore
- Azure Disk, Files

**Shared storage for distributed training:**
- NFS
- Ceph
- Cloud-native solutions (EFS, Filestore)

### 35. Helm charts cho ML applications?

**Benefits:**
- Packaging ML applications
- Templating và values.yaml
- Chart dependencies
- Release management
- Version control

**Example values.yaml:**
```yaml
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

### 36. Namespaces và multi-tenancy cho ML teams?

**Use cases:**
- Isolation giữa teams/projects
- Resource quotas per namespace
- RBAC (Role-Based Access Control)
- Network policies
- Environment separation (dev, staging, prod)

**Example ResourceQuota:**
```yaml
apiVersion: v1
kind: ResourceQuota
metadata:
  name: ml-team-quota
  namespace: ml-team
spec:
  hard:
    requests.cpu: "100"
    requests.memory: 200Gi
    requests.nvidia.com/gpu: "4"
    persistentvolumeclaims: "10"
```

### 37. Monitoring ML models trên Kubernetes?

**Monitoring stack:**
- **Prometheus:** metrics collection
- **Grafana:** dashboards
- **EFK stack:** logging (Elasticsearch, Fluentd, Kibana)

**Metrics to track:**
- Infrastructure: CPU, memory, GPU utilization
- Application: request rate, latency, errors
- ML-specific: prediction distribution, confidence scores
- Custom metrics: model accuracy, drift

### 38. Canary deployments và A/B testing cho models?

**Istio VirtualService example:**
```yaml
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

**Strategies:**
- Canary: gradual rollout (5% → 20% → 50% → 100%)
- A/B testing: user segmentation
- Blue-Green: instant switch
- Shadow: parallel without affecting users

### 39. Distributed training trên Kubernetes?

**Frameworks:**
- **Kubeflow Training Operators:**
  - TFJob (TensorFlow)
  - PyTorchJob (PyTorch)
  - MXNetJob (MXNet)
- **Horovod với MPI Operator**
- **Ray on Kubernetes**

**Architecture:**
- Parameter servers và workers
- Multi-node GPU training
- Shared storage for data
- Communication optimization

### 40. Service Mesh (Istio) benefits cho ML services?

**Benefits:**
- **Traffic management:**
  - Canary deployments
  - A/B testing
  - Traffic splitting
- **Security:**
  - mTLS between services
  - Authorization policies
- **Observability:**
  - Distributed tracing
  - Metrics collection
- **Reliability:**
  - Circuit breaking
  - Retries và timeouts
  - Fault injection for testing

---

[← CI/CD](./03-cicd.md) | [MLOps Tools →](./05-mlops-tools.md)
