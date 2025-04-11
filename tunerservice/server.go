/*
Handles API endpoints and HTTP server logic.
*/

package tunerservice

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.ibm.com/ai-platform-optimization/inferno/services/controller"
)

var TS *TunerService
var Wg sync.WaitGroup

type TunerServer struct {
	router *gin.Engine
}

func NewTunerServer() (*TunerServer, error) {
	ts := &TunerServer{
		router: gin.Default(),
	}
	ts.router.GET("/getparams", getparams)
	CollectorURL = controller.GetURL(CollectorHostEnvName, CollectorPortEnvName)
	return ts, nil
}

func (ts *TunerServer) Run(tunerPeriod int) {
	// Start the server
	var err error
	TS, err = NewTunerService()
	if err != nil {
		fmt.Printf("Error creating tuner service")
		return
	}

	Wg.Add(1)
	go func() {
		defer Wg.Done()
		host := "localhost"
		port := "8080"

		if h := os.Getenv("TUNER_HOST"); h != "" {
			host = h
		}
		if p := os.Getenv("TUNER_PORT"); p != "" {
			port = p
		}
		ts.router.Run(host + ":" + port)
	}()

	// also start periodic environment updates
	if tunerPeriod > 0 {
		Wg.Add(1)
		go func() {
			defer Wg.Done()
			agentTicker := time.NewTicker(time.Second * time.Duration(tunerPeriod))
			defer agentTicker.Stop()

			for range agentTicker.C {
				if err := TS.UpdateTunersAndRun(); err != nil {
					fmt.Printf("%v: skipping cycle ... reason=%s\n", time.Now().Format("15:04:05.000"), err.Error())
				}
			}
		}()
	}
}
