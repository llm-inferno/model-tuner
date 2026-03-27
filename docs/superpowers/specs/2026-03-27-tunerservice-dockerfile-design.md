# Dockerfile for tunerservice — Design Spec

**Date:** 2026-03-27
**Status:** Approved

## Overview

Add a `Dockerfile` at the repo root to build and run the `tunerservice` REST API server as a container image, suitable for Kubernetes pod deployment.

## Approach

Multi-stage build with an Alpine runtime image. Chosen over distroless for easier debugging during the experimental phase (shell access via `kubectl exec`).

## Build Stage

- Base image: `golang:1.25-alpine`
- `WORKDIR /build`
- Copy `go.mod` + `go.sum` first, run `go mod download` — caches the dependency layer so code changes don't trigger a re-download
- Copy full source tree
- Build: `CGO_ENABLED=0 GOOS=linux go build -o tunerservice ./demos/tunerservice`

## Runtime Stage

- Base image: `alpine:3`
- `WORKDIR /app`
- Copy binary `/build/tunerservice` from builder stage
- Copy `config-data/` into image at `/app/config-data/` (baked-in default configuration fallback)
- `ENV CONFIG_DATA_DIR=/app/config-data` — points to baked-in config by default
- `ENV TUNER_HOST=0.0.0.0` — overrides the `localhost` default so the server binds to all interfaces inside the pod
- `EXPOSE 8081`
- `ENTRYPOINT ["/app/tunerservice"]`

## ConfigMap Mount Strategy

- At deploy time, Kubernetes mounts the ConfigMap to a path such as `/etc/tuner/config-data/` and sets `CONFIG_DATA_DIR=/etc/tuner/config-data` via an environment variable override
- If no ConfigMap is mounted, `CONFIG_DATA_DIR` remains `/app/config-data/` and the baked-in defaults are used
- No fallback logic is needed in the Dockerfile or application code — the env var determines which config is active

## Files to Create

- `Dockerfile` at repo root
- `.dockerignore` at repo root (exclude `docs/`, `.git/`, demo binaries, etc.)

## Out of Scope

- Kubernetes manifests / Helm charts
- Docker Compose
- CI/CD pipeline changes
