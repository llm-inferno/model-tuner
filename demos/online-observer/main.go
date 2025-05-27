package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"os"
	"os/signal"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/core"
	"github.ibm.com/modeling-analysis/model-tuner/pkg/utils"
)

type record struct {
	RPM          float32
	Tokens       float32
	MaxBatchSize int
	AvgBatchSize float32
	AvgWait      float32
	AvgItl       float32
	Alpha        float64
	Beta         float64
	DiffWait     float64
	DiffITL      float64

	// Experiment metadata
	ModelName     string
	ExperimentRPM int
	InputLen      int
	OutputLen     int
	Dataset       string
}

func main() {
	// collect experiment metadata for records
	scanner := bufio.NewScanner(os.Stdin)

	fmt.Print("Enter model name (default: facebook/opt-125m): ")
	scanner.Scan()
	modelName := strings.TrimSpace(scanner.Text())
	if modelName == "" {
		modelName = "facebook/opt-125m"
	}

	fmt.Print("Enter dataset name (default: random): ")
	scanner.Scan()
	dataset := strings.TrimSpace(scanner.Text())
	if dataset == "" {
		dataset = "random"
	}

	fmt.Print("Enter actual request rate (RPM) (optional): ")
	scanner.Scan()
	experimentRPM, _ := strconv.Atoi(strings.TrimSpace(scanner.Text()))

	fmt.Print("Enter input length (optional): ")
	scanner.Scan()
	inputLen, _ := strconv.Atoi(strings.TrimSpace(scanner.Text()))

	fmt.Print("Enter output length (optional): ")
	scanner.Scan()
	outputLen, _ := strconv.Atoi(strings.TrimSpace(scanner.Text()))

	observer, err := core.NewOnlineObserver()
	if err != nil {
		fmt.Printf("Error in Observer creation: %s\n", err)
	}
	configData, err := utils.LoadConfigForServer("default")
	if err != nil {
		fmt.Printf("Error in loading config data: %s\n", err)
	}

	// create tuner by supplying the observers environment
	env := observer.GetEnvironment()
	tuner, err := core.NewTuner(configData, env)
	if err != nil {
		fmt.Printf("Error in creating the tuner: %s\n", err)
		return
	}
	fmt.Println(tuner)

	var records []record
	numSteps := 200

	sigs := make(chan os.Signal, 1)
	done := make(chan struct{}) // block main loop on Ctrl+C
	signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		<-sigs
		fmt.Print("\nCtrl+C detected. Enter filename for CSV output (or press Enter to skip): ")
		var filename string
		fmt.Scanln(&filename)

		filename = strings.TrimSpace(filename)
		if filename == "" {
			fmt.Println("No filename provided. Skipping CSV write.")
		} else {
			if !strings.HasSuffix(strings.ToLower(filename), ".csv") {
				filename += ".csv"
			}
			writeCSV(filename, records)
		}
		close(done) // signal main to exit
	}()

loop:
	for k := 0; k < numSteps; k++ {
		select {
		case <-done:
			break loop // exit loop cleanly
		default:
			// continue regular loop
		}

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

		x := tuner.X().RawVector().Data
		delta := tuner.Innovation().RawVector().Data

		r := record{
			RPM:          env.Lambda,
			Tokens:       env.AvgTokensPerRequest,
			MaxBatchSize: env.MaxBatchSize,
			AvgBatchSize: env.BatchSize,
			AvgWait:      env.AvgQueueTime,
			AvgItl:       env.AvgTokenTime,
			Alpha:        x[0],
			Beta:         x[1],
			DiffWait:     delta[0],
			DiffITL:      delta[1],

			ModelName:     modelName,
			ExperimentRPM: experimentRPM,
			InputLen:      inputLen,
			OutputLen:     outputLen,
			Dataset:       dataset,
		}
		records = append(records, r)

		fmt.Printf("%d : %s;   %s;   %s\n",
			k,
			utils.VecString("X", tuner.X()),
			utils.VecString("Delta", tuner.Innovation()),
			utils.MatString("P", tuner.P()))

		time.Sleep(60 * time.Second)
	}
	fmt.Println("Tuning stopped. Goodbye.")
}

func writeCSV(filename string, data []record) {
	file, err := os.Create(filename)
	if err != nil {
		fmt.Printf("Failed to create CSV file: %s\n", err)
		return
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	header := []string{
		"RPM", "Tokens", "MaxBatchSize", "AvgBatchSize", "AvgWait", "AvgItl",
		"alpha", "beta", "diffWait", "diffITL", "", // empty spacer column
		"Model", "ExperimentRPM", "InputLen", "OutputLen", "Dataset",
	}
	writer.Write(header)

	for _, r := range data {
		row := []string{
			fmt.Sprintf("%.3f", r.RPM),
			fmt.Sprintf("%.3f", r.Tokens),
			strconv.Itoa(r.MaxBatchSize),
			fmt.Sprintf("%.3f", r.AvgBatchSize),
			fmt.Sprintf("%.3f", r.AvgWait),
			fmt.Sprintf("%.3f", r.AvgItl),
			fmt.Sprintf("%.6f", r.Alpha),
			fmt.Sprintf("%.6f", r.Beta),
			fmt.Sprintf("%.6f", r.DiffWait),
			fmt.Sprintf("%.6f", r.DiffITL),
			"", // empty column for spacing
			r.ModelName,
			strconv.Itoa(r.ExperimentRPM),
			strconv.Itoa(r.InputLen),
			strconv.Itoa(r.OutputLen),
			r.Dataset,
		}
		writer.Write(row)
	}
	fmt.Println("CSV file written successfully as", filename)
}
