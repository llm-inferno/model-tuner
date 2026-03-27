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
