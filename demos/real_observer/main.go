package main

import (
	"encoding/json"
	"fmt"
	"os"
	"time"

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

	observer, err := core.NewRealObserver()
	if err != nil {
		fmt.Printf("Error in Observer creation: %s\n", err)
	}

	// env := observer.GetEnvironment()

	// fmt.Println(env.String())

	// create tuner
	tuner, err := core.NewTuner(&configData, observer)
	if err != nil {
		fmt.Println(err)
		return
	}
	fmt.Println(tuner)

	// run tuner a number of steps
	numSteps := 100
	for k := 0; k < numSteps; k++ {
		env := observer.GetEnvironment()

		if env == nil {
			fmt.Println("error getting the environment")
			continue
		}
		fmt.Println(env.String())
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
