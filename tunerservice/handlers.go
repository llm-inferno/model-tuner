package tunerservice

import (
	"net/http"

	"github.com/gin-gonic/gin"
)

// Handlers for REST API calls

func getparams(c *gin.Context) {
	serverName := c.Query("server_name") // Get server name from query params

	if serverName == "" {
		c.JSON(400, gin.H{"error": "server_name query parameter is required"})
		return
	}

	mutex.Lock()
	tuner, exists := TS.Tuners[serverName]
	mutex.Unlock()

	if !exists {
		c.JSON(http.StatusNotFound, gin.H{"error": "server not found"})
		return
	}

	stateVec := tuner.GetParams()
	if stateVec == nil || stateVec.Len() < 2 {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "invalid state vector"})
	}

	alpha := stateVec.AtVec(0)
	beta := stateVec.AtVec(1)

	c.JSON(http.StatusOK, gin.H{
		"server_name": serverName,
		"alpha":       alpha,
		"beta":        beta,
	})

}
