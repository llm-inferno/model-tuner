package tunerservice

import (
	"net/http"

	"github.com/gin-gonic/gin"
	optconfig "github.com/llm-inferno/optimizer-light/pkg/config"
)

// POST /tune
// Request body: []config.ServerSpec (ReplicaSpecs from the control-loop Collector)
// Response:     config.ModelData with updated alpha/beta/gamma per model/accelerator pair
func (ts *TunerServer) handleTune(c *gin.Context) {
	var replicaSpecs []optconfig.ServerSpec
	if err := c.ShouldBindJSON(&replicaSpecs); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
		return
	}
	if len(replicaSpecs) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "replicaSpecs must not be empty"})
		return
	}

	modelData, err := ts.service.Tune(replicaSpecs)
	if err != nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, modelData)
}

// GET /getparams?model=<name>&accelerator=<acc>
// Response: LearnedParameters for the given model/accelerator pair
func (ts *TunerServer) handleGetParams(c *gin.Context) {
	model := c.Query("model")
	accelerator := c.Query("accelerator")

	if err := validateKey(model, accelerator); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}

	params := ts.service.GetParams(model, accelerator)
	if params == nil {
		c.JSON(http.StatusNotFound, gin.H{"error": "no parameters found for model=" + model + " accelerator=" + accelerator})
		return
	}
	c.JSON(http.StatusOK, gin.H{
		"model":       model,
		"accelerator": accelerator,
		"alpha":       params.Alpha,
		"beta":        params.Beta,
		"gamma":       params.Gamma,
		"nis":         params.NIS,
		"lastUpdated": params.LastUpdated,
	})
}
