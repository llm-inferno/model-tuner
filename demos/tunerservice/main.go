package main

import (
	"log"

	"model-tuner/tunerservice"
)

func main() {
	ts, err := tunerservice.NewTunerServer()
	if err != nil {
		log.Fatalf("Failed to initiate TunerService: %v", err)
	}

	tunerPeriod := 60
	log.Println("Starting TunerService...")
	ts.Run(tunerPeriod)
	tunerservice.Wg.Wait()
}
