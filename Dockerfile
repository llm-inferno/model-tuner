# Build stage
FROM golang:1.25-alpine AS builder

WORKDIR /build

# Cache dependency layer separately from source
COPY go.mod go.sum ./
RUN go mod download

# Copy full source and build static binary
COPY . .
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -trimpath -o tunerservice ./cmd/tuner

# Runtime stage
FROM alpine:3.20

# Run as non-root user
RUN addgroup -S tuner && adduser -S -G tuner tuner

WORKDIR /app

# Copy binary and baked-in default config
COPY --from=builder /build/tunerservice .
COPY config-data/ ./config-data/

# Ensure non-root user owns all app files
RUN chown -R tuner:tuner /app

# CONFIG_DATA_DIR points to baked-in defaults; override with a ConfigMap mount at deploy time
ENV CONFIG_DATA_DIR=/app/config-data
# Bind to all interfaces inside the pod (overrides the localhost default)
ENV TUNER_HOST=0.0.0.0
# Suppress Gin debug banner in production
ENV GIN_MODE=release

USER tuner

EXPOSE 8081

ENTRYPOINT ["/app/tunerservice"]
