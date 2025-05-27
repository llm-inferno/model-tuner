package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/core"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/utils"
)

type result struct {
	Step     int
	Alpha    float64
	Beta     float64
	DiffWait float64
	DiffITL  float64
}

func main() {
	prefix := "../../samples/"
	envFilePath := prefix + "mb8-rpm6-a100.csv"
	observer, err := core.NewOfflineObserver(envFilePath)
	if err != nil {
		fmt.Printf("Error in Observer creation: %s\n", err)
	}

	configData, err := utils.LoadConfigForServer("default")
	if err != nil {
		fmt.Printf("Error in loading config data: %s\n", err)
	}

	// create tuner by first getting the environment
	env := observer.GetEnvironment()
	tuner, err := core.NewTuner(configData, env)
	if err != nil {
		fmt.Printf("Error in creating the tuner: %s\n", err)
		return
	}
	fmt.Println(tuner)

	var results []result
	numSteps := 1000

	for k := 0; k < numSteps; k++ {
		env := observer.GetEnvironment()
		if env == nil {
			break
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

		x := tuner.X().RawVector().Data
		delta := tuner.Innovation().RawVector().Data
		results = append(results, result{
			Step:     k,
			Alpha:    x[0],
			Beta:     x[1],
			DiffWait: delta[0],
			DiffITL:  delta[1],
		})
	}
	fmt.Print("Enter output CSV file name (leave blank to skip saving): ")
	var outputFileName string
	fmt.Scanln(&outputFileName)

	if outputFileName != "" {
		err := writeResultsToCSV(outputFileName, results)
		if err != nil {
			fmt.Printf("Failed to write results: %s\n", err)
		} else {
			fmt.Printf("Results written to %s\n", outputFileName)
		}
	}
}

func writeResultsToCSV(filename string, results []result) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Write header
	writer.Write([]string{"Step", "Alpha", "Beta", "DiffWait", "DiffITL"})

	// Write each result row
	for _, r := range results {
		row := []string{
			strconv.Itoa(r.Step),
			fmt.Sprintf("%.6f", r.Alpha),
			fmt.Sprintf("%.6f", r.Beta),
			fmt.Sprintf("%.6f", r.DiffWait),
			fmt.Sprintf("%.6f", r.DiffITL),
		}
		writer.Write(row)
	}
	return nil
}
