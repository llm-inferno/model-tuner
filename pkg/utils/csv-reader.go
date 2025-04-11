package utils

import (
	"encoding/csv"
	"fmt"
	"os"
)

func ReadCsvFile(filePath string) ([][]string, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, fmt.Errorf("unable to read input file %s: %v", filePath, err)
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	records, err := csvReader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("unable to parse file as CSV for %s: %v", filePath, err)
	}
	if len(records) < 2 {
		return nil, fmt.Errorf("not enough data")
	}
	return records, nil
}
