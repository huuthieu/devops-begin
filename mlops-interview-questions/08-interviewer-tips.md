# Tips cho Người Phỏng Vấn

[← Quay lại](./README.md) | [← Additional Topics](./07-additional-topics.md)

## Đánh giá Candidates theo Level

### Junior Level (0-2 years)

**Technical Expectations:**
- ✅ Hiểu basic Docker concepts và có thể viết simple Dockerfile
- ✅ Familiar với Docker Compose cho local development
- ✅ Hiểu CI/CD concepts và đã sử dụng một công cụ (GitHub Actions, GitLab CI)
- ✅ Basic Kubernetes knowledge (Pods, Deployments, Services)
- ✅ Biết một vài ML tools (MLflow hoặc W&B hoặc similar)
- ✅ Đã deploy ít nhất một model to production (có thể simple)

**Câu hỏi phù hợp:**
- Docker & Dockerfile: Câu 1-5
- Docker Compose: Câu 11-14
- CI/CD: Câu 18-20
- Kubernetes: Câu 26-30
- MLOps Tools: Câu 41-46

**Red flags:**
- Không có hands-on experience, chỉ theoretical knowledge
- Chưa bao giờ deploy anything to production
- Không hiểu importance của version control
- Không biết cơ bản về Linux/command line

**Green flags:**
- Side projects với complete deployment
- Curiosity và willingness to learn
- Basic understanding của ML lifecycle
- Good documentation habits

---

### Mid Level (2-4 years)

**Technical Expectations:**
- ✅ Hands-on experience containerizing ML applications
- ✅ Thiết kế và implement CI/CD pipelines cho ML projects
- ✅ Deploy và manage applications trên Kubernetes
- ✅ Experience với experiment tracking và model registry
- ✅ Understanding của complete ML lifecycle
- ✅ Troubleshoot production issues
- ✅ Optimize model inference performance

**Câu hỏi phù hợp:**
- Advanced Docker: Câu 6-10
- Complete ML pipelines: Câu 15-17
- CI/CD implementation: Câu 21-23
- Kubernetes deployments: Câu 31-35
- MLOps tools hands-on: Câu 47-55
- Basic scenarios: Câu 61-65

**Red flags:**
- Cannot explain trade-offs in design decisions
- No experience with monitoring/observability
- Ignore cost considerations
- Poor understanding of ML-specific challenges
- Cannot debug production issues

**Green flags:**
- Production experience với multiple models
- Understanding của trade-offs (cost vs performance)
- Automation mindset
- Monitoring-first approach
- Can discuss failures và lessons learned
- Contribution to team process improvements

**Sample Interview Flow:**
1. **Technical deep-dive (45 min):**
   - Walk through một production deployment
   - Discuss CI/CD pipeline they built
   - Troubleshoot a scenario (câu 61-70)

2. **System design (30 min):**
   - Design ML serving system for specific requirements
   - Discuss trade-offs

3. **Cultural fit (15 min):**
   - How they handle incidents
   - Collaboration with data scientists
   - Learning approach

---

### Senior Level (4+ years)

**Technical Expectations:**
- ✅ Architect end-to-end MLOps platforms
- ✅ Advanced Kubernetes (operators, custom resources, service mesh)
- ✅ Design distributed training systems
- ✅ Implement comprehensive monitoring và alerting
- ✅ Cost optimization strategies
- ✅ Security best practices
- ✅ Mentoring junior engineers
- ✅ Drive best practices adoption

**Câu hỏi phù hợp:**
- System architecture questions
- Advanced Kubernetes: Câu 36-40
- Infrastructure as Code: Câu 53-54
- Security & compliance: Câu 58-60
- Complex scenarios: Câu 66-70
- Platform design: Câu 71-75

**Red flags:**
- Cannot discuss architecture trade-offs deeply
- No experience with scale (high traffic, large models)
- Ignore security considerations
- Cannot mentor or explain concepts clearly
- Rigid thinking, not adaptable
- No strategic thinking about platform evolution

**Green flags:**
- Built or significantly contributed to ML platform
- Deep understanding of trade-offs (cost, performance, reliability)
- Security-conscious by default
- Can discuss multiple approaches với pros/cons
- Experience with incidents và post-mortems
- Contribution to open source hoặc technical blogs
- Can translate business requirements to technical solutions
- Mentoring experience

**Sample Interview Flow:**
1. **System design (60 min):**
   ```
   "Design an MLOps platform for a company with:
   - 50 data scientists
   - 100+ models in production
   - 10M requests/day
   - Multi-region deployment
   - Strict compliance requirements"
   ```

2. **Technical leadership (30 min):**
   - How to introduce new technology
   - Handling technical debt
   - Building team capabilities
   - Incident management

3. **Architecture review (30 min):**
   - Review current architecture (if available)
   - Suggest improvements
   - Discuss trade-offs

---

## Đánh giá Kỹ năng Cụ thể

### Docker & Containerization

**Junior:**
- Can write basic Dockerfile
- Understand images vs containers
- Use docker-compose for local dev

**Mid:**
- Multi-stage builds
- Optimize image size
- Understand layer caching
- Security basics (non-root user)

**Senior:**
- Advanced optimization techniques
- Security hardening
- Custom base images
- Registry management

**Assessment:**
```
Give them a simple Python ML app and ask them to:
1. Containerize it
2. Optimize the image
3. Make it production-ready
```

---

### Kubernetes

**Junior:**
- Basic concepts (Pods, Services, Deployments)
- Can deploy using kubectl
- Understand YAML configs

**Mid:**
- Resource management
- Auto-scaling
- ConfigMaps/Secrets
- Debugging issues

**Senior:**
- Custom resources và operators
- Service mesh
- Multi-tenancy
- Disaster recovery

**Assessment:**
```
Scenario: "Your model serving pods are being OOM killed.
How do you:
1. Diagnose the issue
2. Fix it
3. Prevent it in the future"
```

---

### CI/CD

**Junior:**
- Understand CI/CD concepts
- Basic pipeline (lint, test, build)
- Used at least one CI/CD tool

**Mid:**
- Design complete ML pipelines
- Data validation gates
- Model testing strategies
- Deployment automation

**Senior:**
- Continuous training pipelines
- Multi-environment strategies
- GitOps implementation
- Pipeline optimization

**Assessment:**
```
"Design a CI/CD pipeline for a model that needs to:
- Retrain weekly
- Deploy only if accuracy > 90%
- Support A/B testing
- Have rollback capability"
```

---

## Red Flags (All Levels)

### Technical Red Flags:
- ❌ Không có hands-on experience, chỉ theoretical
- ❌ Cannot explain past decisions
- ❌ Ignore reproducibility importance
- ❌ No monitoring/observability mindset
- ❌ Ignore security completely
- ❌ No version control discipline
- ❌ Cannot troubleshoot issues
- ❌ Overengineering simple problems

### Behavioral Red Flags:
- ❌ Blame others for failures
- ❌ Not curious about learning
- ❌ Cannot work with data scientists
- ❌ Rigid, not adaptable
- ❌ Poor communication
- ❌ No ownership mindset

---

## Green Flags (All Levels)

### Technical Green Flags:
- ✅ Production experience
- ✅ Understanding của trade-offs
- ✅ Automation mindset
- ✅ Monitoring-first approach
- ✅ Security awareness
- ✅ Cost consciousness
- ✅ Good documentation habits
- ✅ Reproducibility focus

### Behavioral Green Flags:
- ✅ Learns from failures
- ✅ Collaborative with data scientists
- ✅ Proactive problem solving
- ✅ Clear communication
- ✅ Ownership mindset
- ✅ Shares knowledge
- ✅ Pragmatic approach

---

## Interview Structure Recommendations

### Technical Screen (60 min)

**Part 1: Experience Discussion (20 min)**
```
- Walk through most complex project
- What was your role?
- What challenges did you face?
- How did you solve them?
- What would you do differently?
```

**Part 2: Technical Questions (25 min)**
```
- 2-3 questions from relevant sections
- Mix of breadth and depth
- Allow discussion of trade-offs
```

**Part 3: Scenario (15 min)**
```
- Give them a production scenario
- How would they approach it?
- What questions would they ask?
```

---

### System Design (60 min)

**Sample Problem:**
```
"Design a system to serve 100 ML models with:
- 1M requests/day total
- 99.9% availability SLA
- <100ms p95 latency
- Auto-scaling based on traffic
- Model versioning and rollback
- Cost optimization"

Evaluate:
- Architecture decisions
- Trade-off discussions
- Scalability considerations
- Monitoring approach
- Cost awareness
- Security thinking
```

---

### Coding/Practical (60-90 min)

**Option 1: Take-home Assignment**
```
Build a simple ML serving API with:
- Containerized application
- CI/CD pipeline
- Kubernetes deployment configs
- Monitoring setup
- Documentation

Evaluate:
- Code quality
- Documentation
- Best practices
- Completeness
```

**Option 2: Live Coding**
```
- Debug a broken deployment
- Write a Dockerfile
- Fix a CI/CD pipeline
- Troubleshoot K8s issues

Evaluate:
- Problem-solving approach
- Tool knowledge
- Debugging skills
- Communication
```

---

## Calibration Guidelines

### Scoring Framework

**For each area, score 1-5:**
- 1: No knowledge/experience
- 2: Basic theoretical knowledge
- 3: Some practical experience
- 4: Strong practical experience
- 5: Expert level

**Areas to evaluate:**
1. Containerization (Docker)
2. Orchestration (Kubernetes)
3. CI/CD
4. ML Tools (MLflow, etc.)
5. Monitoring & Observability
6. System Design
7. Problem Solving
8. Communication

**Hiring Bar:**
- Junior: Average ≥ 2.5, no area < 2
- Mid: Average ≥ 3.5, no area < 3
- Senior: Average ≥ 4.0, no area < 3, at least 3 areas = 5

---

## Common Interview Mistakes to Avoid

### Interviewer Mistakes:
- ❌ Asking only theoretical questions
- ❌ Not allowing discussion of trade-offs
- ❌ Expecting perfect answers
- ❌ Ignoring problem-solving approach
- ❌ Not checking hands-on experience
- ❌ Too narrow focus on one technology

### Best Practices:
- ✅ Focus on problem-solving approach
- ✅ Allow candidates to ask questions
- ✅ Evaluate trade-off discussions
- ✅ Check hands-on experience
- ✅ Give realistic scenarios
- ✅ Assess learning ability

---

## Sample Questions by Category

### Problem Solving:
```
"A model's accuracy dropped from 95% to 75% overnight.
Walk me through how you would investigate this."
```

### Trade-offs:
```
"When would you use batch prediction vs online prediction?
Discuss the trade-offs."
```

### System Design:
```
"Design a feature store for a team of 20 data scientists.
What components would you include and why?"
```

### Practical:
```
"This Dockerfile builds slowly. How would you optimize it?"
```

---

## Post-Interview

### Documentation:
- Record detailed feedback for each area
- Specific examples of strengths/weaknesses
- Recommendation với reasoning
- Compare with team bar

### Feedback to Candidate:
- Be specific về areas to improve
- Highlight strengths
- Provide resources for learning
- Encourage reapplication if close

---

**Remember:** The goal is to find candidates who:
1. Can solve real problems
2. Understand trade-offs
3. Will grow with the team
4. Collaborate well
5. Have strong fundamentals

Not perfect knowledge of every tool!

---

[← Additional Topics](./07-additional-topics.md) | [← Quay lại](./README.md)
