package estimator

// baseFactor is the fraction of ITL assumed to be baseline iteration overhead (alpha).
// Used in GuessInitState to derive an initial estimate of alpha from observed ITL.
const baseFactor = 0.9
