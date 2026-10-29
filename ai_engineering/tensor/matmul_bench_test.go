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
	kernels := []struct {
		name string
		f    func(a, b, out []float32, n, k, p int)
	}{
		{"naive", MatMulNaive}, {"naive4", MatMulNaive4}, {"ikj", MatMulIKJ},
		{"tiled", MatMulTiled}, {"parallel", MatMulParallel},
	}
	for _, n := range []int{128, 512, 1024} {
		a, bm := Randn(rng, n, n).Data(), Randn(rng, n, n).Data()
		for _, k := range kernels {
			b.Run(fmt.Sprintf("%s/n=%d", k.name, n), func(b *testing.B) {
				out := make([]float32, n*n)
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
