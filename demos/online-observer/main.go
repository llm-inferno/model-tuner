package main

import (
	"fmt"
	"time"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/core"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/utils"
)

func main() {
	observer, err := core.NewOnlineObserver()
	if err != nil {
		fmt.Printf("Error in Observer creation: %s\n", err)
	}
	configData, err := utils.LoadConfigForServer("default")
	if err != nil {
		fmt.Printf("Error in loading config data: %s\n", err)
	}

	// create tuner by supplying the appropriate environment
	env := observer.GetEnvironment()
	tuner, err := core.NewTuner(configData, env)
	if err != nil {
		fmt.Printf("Error in creating the tuner: %s\n", err)
		return
	}
	fmt.Println(tuner)

	// run tuner a number of steps
	numSteps := 200
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

		time.Sleep(60 * time.Second)
	}
}
