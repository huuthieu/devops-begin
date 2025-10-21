# CI/CD

[← Quay lại](./README.md) | [← Docker Compose](./02-docker-compose.md) | [Kubernetes →](./04-kubernetes.md)

## Câu hỏi cơ bản

### 18. CI/CD là gì và tại sao quan trọng trong MLOps?

**Key points:**
- Continuous Integration: automated testing, building
- Continuous Deployment: automated deployment
- Benefits: faster iteration, reproducibility, quality assurance
- Automation trong ML lifecycle

### 19. Các stages chính trong ML CI/CD pipeline?

**Pipeline stages:**
1. Code quality checks (linting, formatting)
2. Unit tests
3. Data validation
4. Model training
5. Model evaluation
6. Model testing (integration, performance)
7. Model deployment
8. Monitoring

### 20. Sự khác biệt giữa traditional CI/CD và ML CI/CD?

**ML-specific aspects:**
- Data versioning
- Model versioning
- Data validation stages
- Model evaluation gates
- A/B testing và gradual rollouts
- Model monitoring
- Retraining triggers

---

## Câu hỏi nâng cao

### 21. Thiết kế một complete CI/CD pipeline cho ML project (GitHub Actions)?

**Complete workflow:**
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

### 22. Continuous Training (CT) pipeline design?

**Key components:**
- Trigger mechanisms:
  - Schedule-based (daily, weekly)
  - Data drift detection
  - Performance degradation
  - Manual trigger
- Automated retraining workflow
- Model comparison và approval gates
- Automated deployment của better models
- Rollback mechanisms

### 23. Model testing strategies trong CI/CD?

**Testing pyramid cho ML:**
1. **Unit tests:**
   - Preprocessing functions
   - Feature engineering
   - Data validation logic

2. **Integration tests:**
   - Full pipeline execution
   - API endpoints
   - Database connections

3. **Model tests:**
   - Performance thresholds (accuracy, F1, etc.)
   - Inference time benchmarks
   - Model size checks
   - Backward compatibility

4. **Data tests:**
   - Schema validation
   - Data drift detection
   - Quality checks

5. **Bias & Fairness tests:**
   - Model fairness metrics
   - Demographic parity
   - Equal opportunity

### 24. GitOps cho ML deployments?

**Principles:**
- Infrastructure as Code (Terraform, CloudFormation)
- Declarative deployments
- Version control for all configs
- Automated sync between Git và cluster state
- Pull-based deployments

**Tools:**
- ArgoCD
- Flux
- Jenkins X

### 25. Các công cụ CI/CD phổ biến cho MLOps?

**General purpose:**
- GitHub Actions
- GitLab CI/CD
- Jenkins
- CircleCI
- Azure DevOps

**ML-specific:**
- Argo Workflows
- Kubeflow Pipelines
- MLflow Projects
- DVC Pipelines
- Kedro

**Comparison considerations:**
- Integration với ML tools
- Scalability
- Cost
- Learning curve
- Community support

---

[← Docker Compose](./02-docker-compose.md) | [Kubernetes →](./04-kubernetes.md)
