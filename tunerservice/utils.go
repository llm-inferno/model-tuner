package tunerservice

import "fmt"

// validateKey is used in handler input validation.
func validateKey(model, accelerator string) error {
	if model == "" {
		return fmt.Errorf("model is required")
	}
	if accelerator == "" {
		return fmt.Errorf("accelerator is required")
	}
	return nil
}
