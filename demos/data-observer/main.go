package main

import (
	"fmt"

	"github.com/llm-inferno/model-tuner/pkg/core"
	"github.com/llm-inferno/model-tuner/pkg/observer"
	"github.com/llm-inferno/model-tuner/pkg/utils"
	"gonum.org/v1/gonum/mat"
)

func main() {

	configData, err := utils.LoadConfigForServer("prefill-decode")
	if err != nil {
		fmt.Printf("Error in loading config data: %s\n", err)
	}

	// collected data
	inputTokens := []float32{254, 209, 235, 225, 228, 216, 249, 222, 238, 224, 230, 219, 252, 230, 230, 241, 237, 231, 233, 236, 242, 250, 251, 240, 233, 239, 225, 226, 252, 228}
	outputTokens := []float32{458, 465, 462, 471, 486, 463, 431, 468, 457, 458, 477, 452, 434, 449, 463, 435, 456, 446, 489, 422, 470, 443, 472, 408, 472, 391, 474, 504, 445, 464}
	rpmTotal := []float32{407.33, 401.34, 418.11, 468.43, 404.94, 428.9, 803.88, 847.01, 909.21, 874.35, 833.4, 821.85, 1573.76, 1666.95, 1689.79, 1657.5, 1490.44, 1538.2, 858.56, 727.25, 857.08, 789.01, 809.23, 811.91, 395, 334.25, 438.48, 433.69, 433.69, 419.31}
	maxBatchSize := []int{512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512}
	numReplicas := []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1}
	itl := []float32{9.82, 9.49, 9.5, 9.84, 9.54, 9.67, 12.65, 12.23, 11.91, 9.75, 9.81, 9.63, 12.77, 12.7, 12.58, 9.89, 10.01, 9.67, 8.94, 9.55, 9.95, 9.71, 9.83, 9.65, 8.85, 9.38, 9.94, 9.9, 9.73, 9.86}
	ttft := []float32{20.19, 18.74, 19.35, 19.51, 19.08, 18.9, 24.53, 23.44, 23.55, 19.23, 19.88, 18.99, 24.95, 24.69, 25.03, 20.08, 20.08, 17.91, 18.38, 18.67, 22.22, 19.86, 19.84, 19.47, 17.98, 18.84, 19.98, 19.81, 19.84, 19.46}
	avgBatchSize := []float32{30.67, 29.64, 30.72, 36.34, 31.42, 32.14, 73.38, 81.13, 82.84, 32.68, 32.64, 29.94, 73.01, 79.56, 82.37, 29.85, 28.47, 27.76, 15.70, 24.54, 33.56, 28.41, 31.42, 26.77, 13.81, 20.54, 34.58, 36.21, 31.44, 32.11}
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
