/*
Offline Observer gets environment values from a file
*/

package core

import (
	"fmt"
	"strconv"

	"github.com/llm-inferno/model-tuner/pkg/utils"
)

type OfflineObserver struct {
	BaseObserver
	Data      [][]string
	CurrIndex int
}

func NewOfflineObserver(envFilePath string) (*OfflineObserver, error) {
	rows, err := utils.ReadCsvFile(envFilePath)
	if err != nil {
		return nil, err
	}

	return &OfflineObserver{
		Data:      rows[1:],
		CurrIndex: 0,
	}, nil
}

func (o *OfflineObserver) GetEnvironment() *Environment {
	if o.CurrIndex >= len(o.Data) {
		fmt.Println("Warning: No more data to read the environement")
		return nil
	}

	row := o.Data[o.CurrIndex]
	o.CurrIndex++

	lambda, _ := strconv.ParseFloat(row[0], 32)
	avgTokens, _ := strconv.ParseFloat(row[1], 32)
	maxBatchSize, _ := strconv.ParseFloat(row[2], 32)
	batchSize, _ := strconv.ParseFloat(row[3], 32)
	avgQueueTime, _ := strconv.ParseFloat(row[4], 32)
	avgTokenTime, _ := strconv.ParseFloat(row[5], 32)

	env := &Environment{
		Lambda:              float32(lambda),
		AvgTokensPerRequest: float32(avgTokens),
		MaxBatchSize:        int(maxBatchSize), // Fixed
		BatchSize:           float32(batchSize),
		AvgQueueTime:        float32(avgQueueTime),
		AvgTokenTime:        float32(avgTokenTime),
	}
	return env
}
