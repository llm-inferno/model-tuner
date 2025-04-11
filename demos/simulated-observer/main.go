package main

import (
	"fmt"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/core"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/utils"
)

func main() {

	configData, err := utils.LoadConfigForServer("default")
	if err != nil {
		fmt.Printf("Error in loading config data: %s\n", err)
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
	env := observer.GetEnvironment()
	tuner, err := core.NewTuner(configData, env)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tuner)

	// run tuner a number of steps
	numSteps := phase1 + phase2 + phase3
	for k := 0; k < numSteps; k++ {
		env := observer.GetEnvironment()
		if env == nil {
			fmt.Println("error getting the environment")
			continue
		}
		fmt.Println(env.String())
		tuner.UpdateEnvironment(env)

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
