//go:build !darwin || !cgo

package tensor

// matMulBLAS and matMulMetal are nil without Accelerate/Metal; the benchmark skips them.
var (
	matMulBLAS  func(a, b, out []float32, n, k, p int)
	matMulMetal func(a, b, out []float32, n, k, p int)
)
