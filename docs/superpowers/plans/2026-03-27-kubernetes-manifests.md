# Kubernetes Manifests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three Kubernetes manifest files under `deploy/` to deploy the model-tuner tunerservice as a ClusterIP-accessible REST API in a Kubernetes cluster.

**Architecture:** Three separate files, each with one responsibility — ConfigMap holds the default EKF config, Deployment runs the container with the ConfigMap mounted as a volume, Service exposes it cluster-internally on port 8081. Applying `kubectl apply -f deploy/` brings up the full stack.

**Tech Stack:** Kubernetes YAML manifests (ConfigMap, Deployment, Service).

---

### Task 1: Create `deploy/configmap.yaml`

**Files:**
- Create: `deploy/configmap.yaml`

- [ ] **Step 1: Create the `deploy/` directory and write the ConfigMap**

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: model-tuner-config
data:
  default-config-data.json: |
    {
        "filterData": {
            "gammaFactor": 1.0,
            "errorLevel": 0.05,
            "tPercentile": 1.96
        },
        "modelData": {
            "initState": [
                10.0,
                0.02,
                0.0001
            ],
            "percentChange": [
                0.25,
                0.25,
                0.25
            ],
            "boundedState": true,
            "minState": [
                4.0,
                0.002,
                0.00001
            ],
            "maxState": [
                100,
                0.2,
                0.001
            ],
            "expectedObservations": [
                200.0,
                40.0
            ]
        }
    }
```

- [ ] **Step 2: Verify the file is valid YAML**

```bash
python3 -c "import yaml, sys; yaml.safe_load(open('deploy/configmap.yaml'))" && echo "valid"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add deploy/configmap.yaml
git commit -m "add Kubernetes ConfigMap for model-tuner default config"
```

---

### Task 2: Create `deploy/deployment.yaml`

**Files:**
- Create: `deploy/deployment.yaml`

- [ ] **Step 1: Write the Deployment**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: model-tuner
spec:
  replicas: 1
  selector:
    matchLabels:
      app: model-tuner
  template:
    metadata:
      labels:
        app: model-tuner
    spec:
      containers:
        - name: model-tuner
          image: quay.io/atantawi/inferno-model-tuner:latest
          ports:
            - containerPort: 8081
          env:
            - name: CONFIG_DATA_DIR
              value: /etc/tuner/config
          volumeMounts:
            - name: config
              mountPath: /etc/tuner/config
              readOnly: true
          resources:
            requests:
              cpu: 100m
              memory: 256Mi
            limits:
              cpu: 500m
              memory: 512Mi
          securityContext:
            runAsNonRoot: true
      volumes:
        - name: config
          configMap:
            name: model-tuner-config
```

- [ ] **Step 2: Verify the file is valid YAML**

```bash
python3 -c "import yaml, sys; yaml.safe_load(open('deploy/deployment.yaml'))" && echo "valid"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add deploy/deployment.yaml
git commit -m "add Kubernetes Deployment for model-tuner"
```

---

### Task 3: Create `deploy/service.yaml`

**Files:**
- Create: `deploy/service.yaml`

- [ ] **Step 1: Write the Service**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: model-tuner
spec:
  type: ClusterIP
  selector:
    app: model-tuner
  ports:
    - port: 8081
      targetPort: 8081
```

- [ ] **Step 2: Verify the file is valid YAML**

```bash
python3 -c "import yaml, sys; yaml.safe_load(open('deploy/service.yaml'))" && echo "valid"
```

Expected: `valid`

- [ ] **Step 3: Commit**

```bash
git add deploy/service.yaml
git commit -m "add Kubernetes Service for model-tuner"
```

---

### Task 4: Verify the full manifest set with dry-run

**Files:** none (validation only)

- [ ] **Step 1: Dry-run apply all three manifests**

```bash
kubectl apply --dry-run=client -f deploy/
```

Expected output (order may vary):
```
configmap/model-tuner-config configured (dry run)
deployment.apps/model-tuner configured (dry run)
service/model-tuner configured (dry run)
```

If `kubectl` is not available, validate with `kubeconform` or skip this step and note it.

- [ ] **Step 2: Commit nothing** (validation only, no code changes)
