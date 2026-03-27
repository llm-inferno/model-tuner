# tunerservice Dockerfile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-ready `Dockerfile` and `.dockerignore` at the repo root to build and run the tunerservice REST API server as an Alpine-based container image.

**Architecture:** Multi-stage build — `golang:1.25-alpine` compiles a static binary from `./demos/tunerservice`, then `alpine:3` provides the minimal runtime. Default config is baked in at `/app/config-data/`; a Kubernetes ConfigMap overrides it via the `CONFIG_DATA_DIR` env var.

**Tech Stack:** Docker multi-stage build, Go 1.25, Alpine Linux, Gin HTTP server.

---

### Task 1: Create `.dockerignore`

**Files:**
- Create: `.dockerignore`

- [ ] **Step 1: Create `.dockerignore` at repo root**

```
.git
.gitignore
docs/
*.md
```

- [ ] **Step 2: Commit**

```bash
git add .dockerignore
git commit -m "add .dockerignore for tunerservice image"
```

---

### Task 2: Create `Dockerfile`

**Files:**
- Create: `Dockerfile`

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
# Build stage
FROM golang:1.25-alpine AS builder

WORKDIR /build

# Cache dependency layer separately from source
COPY go.mod go.sum ./
RUN go mod download

# Copy full source and build static binary
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -o tunerservice ./demos/tunerservice

# Runtime stage
FROM alpine:3

WORKDIR /app

# Copy binary and baked-in default config
COPY --from=builder /build/tunerservice .
COPY config-data/ ./config-data/

# CONFIG_DATA_DIR points to baked-in defaults; override with a ConfigMap mount at deploy time
ENV CONFIG_DATA_DIR=/app/config-data
# Bind to all interfaces inside the pod (overrides the localhost default)
ENV TUNER_HOST=0.0.0.0

EXPOSE 8081

ENTRYPOINT ["/app/tunerservice"]
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile
git commit -m "add Dockerfile for tunerservice REST API server"
```

---

### Task 3: Verify the image builds and runs

**Files:** none (validation only)

- [ ] **Step 1: Build the image**

```bash
docker build -t tunerservice:local .
```

Expected: build completes with no errors, final image is based on `alpine:3`.

- [ ] **Step 2: Run the container and check it starts**

```bash
docker run --rm -p 8081:8081 tunerservice:local
```

Expected output (within a few seconds):
```
{"time":"...","level":"INFO","msg":"starting TunerServer","addr":"0.0.0.0:8081"}
```

- [ ] **Step 3: Smoke-test the running container (separate terminal)**

```bash
curl -s http://localhost:8081/getparams?model=probe&accelerator=probe
```

Expected: HTTP 200 or 404 JSON response (not a connection error).

- [ ] **Step 4: Verify the ConfigMap override works**

```bash
docker run --rm -p 8081:8081 \
  -v $(pwd)/config-data:/etc/tuner/config \
  -e CONFIG_DATA_DIR=/etc/tuner/config \
  tunerservice:local
```

Expected: server starts as in Step 2, now reading from the mounted directory.

- [ ] **Step 5: Stop the container and commit nothing** (verification task, no code changes)
