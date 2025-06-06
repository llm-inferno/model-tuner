package tunerservice

import "os"

// get URL of a REST server
func GetURL(hostEnvName, portEnvName string) string {
	host := "localhost"
	port := "8080"
	if h := os.Getenv(hostEnvName); h != "" {
		host = h
	}
	if p := os.Getenv(portEnvName); p != "" {
		port = p
	}
	return "http://" + host + ":" + port
}
