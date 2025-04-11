/*
Manages the core tuner server functionality.
*/

package tunerservice

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/core"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/utils"
)

var mutex sync.Mutex

/*
	tuner service acts both as a server and client.

As a client, it uses getenv primitive to get the environment for every server.
As a server, the getparams primitive returns the alpha, beta of the asked server.
*/
type TunerService struct {
	Tuners map[string]*core.Tuner
}

func NewTunerService() (*TunerService, error) {
	ts := &TunerService{
		Tuners: map[string]*core.Tuner{},
	}
	return ts, nil
}

func (ts *TunerService) UpdateTunersAndRun() error {
	// fmt.Println("\n==================== Fetching Environments and Updating Tuners ====================")
	envs, err := ts.GetAllEnvironments()
	if err != nil {
		return fmt.Errorf("error fetching environments: %w", err)
	}

	for serverName, env := range envs {
		mutex.Lock()
		tuner, exists := ts.Tuners[serverName]
		mutex.Unlock()

		// if tuner exists, just update the environment
		if exists {
			tuner.UpdateEnvironment(env)
		} else {
			// Otherwise, create a new tuner
			configData, err := utils.LoadConfigForServer("default")
			if err != nil {
				return fmt.Errorf("error fetching config for server %s: %v", serverName, err)
			}

			tuner, err = core.NewTuner(configData, env)
			if err != nil {
				return fmt.Errorf("error creating tuner for %s: %v", serverName, err)
			}
			mutex.Lock()
			ts.Tuners[serverName] = tuner
			mutex.Unlock()
		}
		fmt.Println(env.String())
		if err := tuner.Run(); err != nil {
			return fmt.Errorf("error running tuner for %s: %v", serverName, err)
		}

		// Print the state of the tuner after update
		fmt.Printf(
			// "Server: %s\n
			"%s;   %s;   %s\n",
			// serverName,
			utils.VecString("X", tuner.X()),
			utils.VecString("Delta", tuner.Innovation()),
			utils.MatString("P", tuner.P()),
		)
	}
	return nil
}

func (ts *TunerService) GetAllEnvironments() (map[string]*core.Environment, error) {
	// call collector to get updated environments for the managed servers
	endPoint := CollectorURL + "/" + CollectEnvVerb
	// fmt.Println("Requesting:", endPoint)

	response, getErr := http.Get(endPoint)
	if getErr != nil {
		return nil, fmt.Errorf("error in getting http response: %v", getErr)
	}
	body, readErr := io.ReadAll(response.Body)
	if readErr != nil {
		return nil, fmt.Errorf("error in reading http response body: %v", readErr)
	}

	envInfo := make(map[string]*core.Environment)
	jsonErr := json.Unmarshal(body, &envInfo)
	if jsonErr != nil {
		return nil, fmt.Errorf("error in unmarshalling json: %v", jsonErr)
	}
	return envInfo, nil
}
