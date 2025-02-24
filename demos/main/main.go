package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/config"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/core"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/utils"
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
	phase1 := 20
	phase2 := 20
	phase3 := 20
	total := phase1 + phase2 + phase3

	rpm := make([]float32, total)
	avgNumTokens := make([]float32, total)
	alpha := make([]float32, total)
	beta := make([]float32, total)
	percentNoise := make([]float32, total)
	maxBatchSize := make([]int, total)

	// phase 1
	for i := 0; i < phase1; i++ {
		rpm[i] = float32(35.2)
		maxBatchSize[i] = 48
		avgNumTokens[i] = float32(1024)
		alpha[i] = float32(18)
		beta[i] = float32(1)
		percentNoise[i] = float32(0.05)
	}
	// phase 2
	for i := phase1; i < phase1+phase2; i++ {
		rpm[i] = float32(35.2)
		maxBatchSize[i] = 48
		avgNumTokens[i] = float32(1024)
		alpha[i] = float32(22)
		beta[i] = float32(0.7)
		percentNoise[i] = float32(0.05)
	}
	// phase 3
	for i := phase1 + phase2; i < phase1+phase2+phase3; i++ {
		rpm[i] = float32(35.2)
		maxBatchSize[i] = 48
		avgNumTokens[i] = float32(1024)
		alpha[i] = float32(18)
		beta[i] = float32(1)
		percentNoise[i] = float32(0.05)
	}

	observer := core.NewSimulatedObserver(rpm, avgNumTokens, alpha, beta, percentNoise, maxBatchSize)
	if observer == nil {
		fmt.Println("invalid parameters for observer")
	}

	// create tuner
	tuner, err := core.NewTuner(&configData, observer)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tuner)

	// run tuner a number of steps
	numSteps := phase1 + phase2 + phase3
	for k := 0; k < numSteps; k++ {
		if err := tuner.Run(); err != nil {
			fmt.Println(err)
			continue
		}
		// print state
		fmt.Printf("%d : %s;   %s;   %s\n",
			k,
			utils.VecString("X", tuner.X()),
			utils.VecString("Delta", tuner.Innovation()),
			utils.MatString("P", tuner.P()))
	}
}
