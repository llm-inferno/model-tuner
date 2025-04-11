package utils

import (
	"encoding/json"
	"fmt"
	"os"

	"github.ibm.com/modeling-analysis/model-tuner/pkg/config"
)

func LoadConfigForServer(serverName string) (*config.ConfigData, error) {
	// Get the directory path from environment variable or use default
	configDir := os.Getenv("CONFIG_DATA_DIR")
	if configDir == "" {
		configDir = "../../samples" // fall back to default directory
	}

	// check if the config data for the server exists, otherwise use default config data
	fileName := fmt.Sprintf("%s/%s-config-data.json", configDir, serverName)
	if _, err := os.Stat(fileName); os.IsNotExist(err) {
		fmt.Printf("Warning: Config for %s not found, using default-config-data.json\n", serverName)
		fileName = fmt.Sprintf("%s/default-config-data.json", configDir)
	}

	// process the json config file
	byteValue, err := os.ReadFile(fileName)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var configData config.ConfigData
	jsonErr := json.Unmarshal(byteValue, &configData)
	if jsonErr != nil {
		return nil, fmt.Errorf("error unmarshalling json data in file %s: %v", fileName, jsonErr)
	}
	return &configData, nil
}
