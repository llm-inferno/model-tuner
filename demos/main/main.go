package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/config"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/core"
)

func main() {

	// get configuration data from file
	prefix := "../../samples/"
	fname := prefix + "config-data.json"
	bytes, err := os.ReadFile(fname)
	if err != nil {
		fmt.Println(err)
		return
	}
	var configData config.ConfigData
	if err := json.Unmarshal(bytes, &configData); err != nil {
		fmt.Println(err)
		return
	}

	// configure simulated system
	rpm := float32(35.2)
	maxBatchSize := 48
	avgNumTokens := float32(1024)
	alpha := float32(19)
	beta := float32(1)
	percentNoise := float32(0.05)
	monitor := core.NewSimulatedObserver(rpm, avgNumTokens, alpha, beta, percentNoise, maxBatchSize)

	// create tuner
	tuner, err := core.NewTuner(&configData, monitor)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tuner)

	// run tuner a number of steps
	numSteps := 20
	tuner.Run(numSteps)
}
