package main

import (
	"fmt"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/observer"
	"github.com/llm-inferno/model-tuner/pkg/utils"
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

	inputTokens := make([]float32, total)
	outputTokens := make([]float32, total)
	alpha := make([]float32, total)
	beta := make([]float32, total)
	gamma := make([]float32, total)
	delta := make([]float32, total)
	percentNoise := make([]float32, total)
	maxBatchSize := make([]int, total)

	// phase 1
	for i := range phase1 {
		rpm[i] = float32(120)
		maxBatchSize[i] = 96
		inputTokens[i] = float32(128)
		outputTokens[i] = float32(512)
		alpha[i] = float32(18)
		beta[i] = float32(0.4)
		gamma[i] = float32(56)
		delta[i] = float32(0.01)
		percentNoise[i] = float32(0.01)
	}
	// phase 2
	for i := phase1; i < phase1+phase2; i++ {
		rpm[i] = float32(60)
		maxBatchSize[i] = 96
		inputTokens[i] = float32(128)
		outputTokens[i] = float32(512)
		alpha[i] = float32(24)
		beta[i] = float32(0.8)
		gamma[i] = float32(72)
		delta[i] = float32(0.02)
		percentNoise[i] = float32(0.01)
	}
	// phase 3
	for i := phase1 + phase2; i < phase1+phase2+phase3; i++ {
		rpm[i] = float32(120)
		maxBatchSize[i] = 96
		inputTokens[i] = float32(128)
		outputTokens[i] = float32(512)
		alpha[i] = float32(18)
		beta[i] = float32(0.4)
		gamma[i] = float32(56)
		delta[i] = float32(0.01)
		percentNoise[i] = float32(0.01)
	}

	observer := observer.NewSimulatedObserver(rpm, inputTokens, outputTokens, alpha, beta, gamma, delta, percentNoise, maxBatchSize)
	if observer == nil {
		fmt.Println("invalid parameters for observer")
	}

	// create tuner
	env := observer.GetEnvironment()
	tuner, _, err := core.SetupTunerForQueueingModel(configData, env, "prefill-decode")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tuner)

	// run tuner a number of steps
	numSteps := phase1 + phase2 + phase3
	for k := range numSteps {
		env = observer.GetEnvironment()
		if env == nil {
			fmt.Println("error getting the environment")
			continue
		}
		fmt.Println(env.String())

		if err := tuner.Run(env); err != nil {
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
