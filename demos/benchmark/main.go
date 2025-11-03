package main

import (
	"fmt"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/observer"
	"github.com/llm-inferno/model-tuner/pkg/utils"
	"gonum.org/v1/gonum/mat"
)

func main() {

	configData, err := utils.LoadConfigForServer("benchmark")
	if err != nil {
		fmt.Printf("Error in loading config data: %s\n", err)
	}

	// collected data
	inputTokens := []float32{64, 64, 64, 64, 64, 64, 64, 64}
	outputTokens := []float32{64, 64, 64, 64, 64, 64, 64, 64}
	rpsTotal := []float32{27.16, 52.06, 76.95, 101.85, 126.73, 151.83, 176.50, 195.95}
	maxBatchSize := []int{512, 512, 512, 512, 512, 512, 512, 512}
	numReplicas := []int{1, 1, 1, 1, 1, 1, 1, 1}
	itl := []float32{7.50, 8.06, 8.63, 9.27, 10.16, 11.47, 15.57, 32.56}
	ttft := []float32{18.05, 19.08, 20.47, 22.01, 24.37, 28.43, 40.56, 99.78}
	avgBatchSize := []float32{13.53, 27.84, 44.08, 62.69, 85.48, 115.82, 183.02, 427.82}

	rpmTotal := make([]float32, len(rpsTotal))
	for i, v := range rpsTotal {
		rpmTotal[i] = v * 60.0
	}

	observer := observer.NewDataObserver(rpmTotal, inputTokens, outputTokens, itl, ttft, avgBatchSize, numReplicas, maxBatchSize)
	if observer == nil {
		fmt.Println("invalid parameters for observer")
	}

	// create tuner
	env := observer.GetEnvironment()
	tuner, observationFuncCreator, err := core.SetupTunerForQueueingModel(configData, env, "prefill-decode")
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tuner)

	// run tuner a number of steps
	numSteps := len(rpmTotal)
	for k := range numSteps {

		fmt.Printf("%d : ", k)

		// evaluate
		var correctedOutput *mat.VecDense
		observationFunc := observationFuncCreator.Create()
		if observationFunc == nil {
			fmt.Println("Error: observation function is nil")
			continue
		}
		if k == 0 {
			nX := len(configData.ModelData.InitState)
			X0 := mat.NewVecDense(nX, configData.ModelData.InitState)
			correctedOutput = observationFunc(X0)
		} else {
			correctedOutput = observationFunc(tuner.X())
		}

		fmt.Print(env.String() + " ; ")

		if err := tuner.Run(env); err != nil {
			fmt.Println(err)
			continue
		}

		fmt.Printf("%s;   %s ; ",
			utils.VecString("X", tuner.X()),
			utils.VecString("Delta", tuner.Innovation()))

		if correctedOutput != nil {
			fmt.Printf("Corrected: %s", utils.VecString("Out", correctedOutput))
		}
		fmt.Println()

		// refresh environment with new data point
		env = observer.GetEnvironment()
		if env == nil {
			fmt.Println("error getting the environment")
			continue
		}
	}
}
