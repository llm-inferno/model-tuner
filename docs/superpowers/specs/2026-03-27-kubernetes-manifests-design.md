# Kubernetes Manifests for model-tuner — Design Spec

**Date:** 2026-03-27
**Status:** Approved

## Overview

Add three Kubernetes manifest files under `deploy/` to deploy the model-tuner tunerservice REST API server to a Kubernetes cluster.

## Files to Create

- `deploy/configmap.yaml`
- `deploy/deployment.yaml`
- `deploy/service.yaml`

Applied together with: `kubectl apply -f deploy/`

## ConfigMap (`deploy/configmap.yaml`)

- **Name:** `model-tuner-config`
- **No namespace** — applied to whatever namespace is active at deploy time
- **Data key:** `default-config-data.json`
- **Data value:** contents of `config-data/default-config-data.json` (the EKF filter and model parameter defaults)
- Serves as the example/fallback config; operators replace or augment it with cluster-specific values

## Deployment (`deploy/deployment.yaml`)

- **Name:** `model-tuner`
- **Replicas:** 1
- **Image:** `quay.io/atantawi/inferno-model-tuner:latest`
- **Container port:** `8081`
- **Labels/selector:** `app: model-tuner`
- **Volume:** ConfigMap `model-tuner-config` mounted at `/etc/tuner/config/` inside the container
- **Environment variable:** `CONFIG_DATA_DIR=/etc/tuner/config` — overrides the baked-in Dockerfile default, directing the service to read config from the mounted ConfigMap
- **Security context:**
  - `runAsNonRoot: true` (Kubernetes enforces this; the Dockerfile already sets `USER tuner`)
- **Resource requests:** `cpu: 100m`, `memory: 256Mi`
- **Resource limits:** `cpu: 500m`, `memory: 512Mi`

## Service (`deploy/service.yaml`)

- **Name:** `model-tuner`
- **Type:** `ClusterIP`
- **Port:** `8081` → `targetPort: 8081`
- **Selector:** `app: model-tuner`
- Other pods in the cluster reach the tuner at `http://model-tuner:8081`

## Out of Scope

- Namespace manifest
- Ingress / external access
- HorizontalPodAutoscaler
- Kustomize overlays (dev/staging/prod)
- RBAC
