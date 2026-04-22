package tunerservice

import (
	"fmt"
	"log/slog"

	"github.com/gin-gonic/gin"
	pkgsvc "github.com/llm-inferno/model-tuner/pkg/service"
)

// TunerServer is the HTTP layer that wraps TunerService and exposes its functionality
// over a Gin REST API.
type TunerServer struct {
	service *pkgsvc.TunerService
	router  *gin.Engine
}

// NewTunerServer creates a TunerServer with the given service and registers all routes.
func NewTunerServer(service *pkgsvc.TunerService) *TunerServer {
	router := gin.Default()
	ts := &TunerServer{service: service, router: router}
	router.POST("/tune", ts.handleTune)
	router.GET("/getparams", ts.handleGetParams)
	router.GET("/warmup", ts.handleWarmUp)
	router.POST("/merge", ts.handleMerge)
	return ts
}

// Run starts the HTTP server on host:port (blocks until the server stops).
func (ts *TunerServer) Run(host, port string) error {
	addr := fmt.Sprintf("%s:%s", host, port)
	slog.Info("starting TunerServer", "addr", addr)
	return ts.router.Run(addr)
}
