package tensor

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// go test ./ai_engineering/tensor -run xxx -bench MatMul -benchtime 3x -count 5
// GFLOP/s = 2·n³ / (ns/op).
func BenchmarkMatMul(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 1))
	type kernel struct {
		name    string
		f       func(a, b, out []float32, n, k, p int)
		aligned bool // run on PageAligned copies of the inputs and output
	}
	kernels := []kernel{
		{name: "naive", f: MatMulNaive}, {name: "naive4", f: MatMulNaive4}, {name: "ikj", f: MatMulIKJ},
		{name: "tiled", f: MatMulTiled}, {name: "parallel", f: MatMulParallel},
	}
	if matMulBLAS != nil { // Accelerate cblas_sgemm, darwin + cgo only
		kernels = append(kernels, kernel{name: "blas", f: matMulBLAS})
	}
	if matMulMetal != nil { // MPSMatrixMultiplication on the Mac GPU
		kernels = append(kernels,
			kernel{name: "metal", f: matMulMetal},
			kernel{name: "metal-nocopy", f: matMulMetal, aligned: true})
	}
	for _, n := range []int{128, 512, 1024, 2048, 4096} {
		aIn, bIn := Randn(rng, n, n).Data(), Randn(rng, n, n).Data()
		for _, k := range kernels {
			b.Run(fmt.Sprintf("%s/n=%d", k.name, n), func(b *testing.B) {
				a, bm, out := aIn, bIn, make([]float32, n*n)
				if k.aligned {
					a, bm, out = PageAligned(n*n), PageAligned(n*n), PageAligned(n*n)
					copy(a, aIn)
					copy(bm, bIn)
				}
				for b.Loop() {
					clear(out)
					k.f(a, bm, out, n, n, n)
				}
				flops := 2 * float64(n) * float64(n) * float64(n)
				b.ReportMetric(flops*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
			})
		}
	}
}
