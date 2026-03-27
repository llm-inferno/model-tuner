package tunerservice

const (
	// baseFactor is the fraction of ITL assumed to be the baseline iteration overhead (alpha).
	// Used in guessInitState to derive an initial estimate of alpha from observed ITL.
	baseFactor = 0.9
)
