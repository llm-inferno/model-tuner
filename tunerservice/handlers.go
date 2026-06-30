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
		"updateCount": params.UpdateCount,
		"lastUpdated": params.LastUpdated,
	})
}

// GET /warmup
// Response: {"warmingUp": bool} — true if any known (model, accelerator) pair still has
// UpdateCount < warmUpCycles; false once all pairs have graduated or warmUpCycles is zero.
func (ts *TunerServer) handleWarmUp(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"warmingUp": ts.service.IsWarmingUp()})
}

// POST /calibrate
// Request body: []config.ServerSpec — a batch of deliberately-diverse sweep operating points
//
//	(benchmarking-on-the-fly), all for one or more (model, accelerator) groups.
//
// Response: config.ModelData with calibrated alpha/beta/gamma per group.
// Returns 422 if no group could be calibrated (e.g. the sweep lacked operating-point spread).
func (ts *TunerServer) handleCalibrate(c *gin.Context) {
	var specs []optconfig.ServerSpec
	if err := c.ShouldBindJSON(&specs); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
		return
	}
	if len(specs) == 0 {
		c.JSON(http.StatusBadRequest, gin.H{"error": "calibration points must not be empty"})
		return
	}

	modelData, err := ts.service.Calibrate(specs)
	if err != nil {
		c.JSON(http.StatusUnprocessableEntity, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, modelData)
}

// GET /calibration-status
// Response: {"statuses": []CalibrationStatus} — per (model, accelerator) pair the tuner has seen,
// the facts the controller's calibration trigger consumes (NeedsCalibration in particular).
func (ts *TunerServer) handleCalibrationStatus(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"statuses": ts.service.CalibrationStatuses()})
}

// POST /merge
// Request body: config.ModelData (the Controller's current ModelData)
// Response:     config.ModelData with PerfParms overlaid from the ParameterStore;
//
//	ParameterStore entries absent from the input are appended with defaults.
func (ts *TunerServer) handleMerge(c *gin.Context) {
	var modelData optconfig.ModelData
	if err := c.ShouldBindJSON(&modelData); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid request body: " + err.Error()})
		return
	}
	merged := ts.service.Merge(&modelData)
	c.JSON(http.StatusOK, merged)
}
