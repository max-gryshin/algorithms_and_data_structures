//go:build race

package train

// The race detector slows the autograd loop ~10–20×; under -race the training tests
// run fewer steps (they still exercise every code path, including the goroutines).
const raceEnabled = true
