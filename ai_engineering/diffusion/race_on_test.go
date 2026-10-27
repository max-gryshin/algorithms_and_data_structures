//go:build race

package diffusion

// The race detector slows the autograd training loop ~20×; under -race the quality
// tests train for a few steps only (to exercise the code paths) and skip the
// quality thresholds, like the standard library does with race.Enabled.
const raceEnabled = true
