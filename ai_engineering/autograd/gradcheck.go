package autograd

import (
	"fmt"
	"math"
)

// GradCheck compares analytic gradients (from Backward) with central finite differences
//
//	∂f/∂θ ≈ (f(θ + ε) − f(θ − ε)) / 2ε
//
// for every element of every param. f must rebuild the graph from params and return
// a scalar loss. It returns the largest relative error
//
//	|analytic − numeric| / max(|analytic| + |numeric|, 1e-3)
//
// and the place where it occurred. With float32 data use eps ≈ 1e-2 and expect
// errors around 1e-3; errors ≳ 1e-1 mean a bug in some backward.
func GradCheck(f func() *Var, params []*Var, eps float64) (maxRel float64, where string) {
	for _, p := range params {
		p.ZeroGrad()
	}
	f().Backward()
	analytic := make([][]float32, len(params))
	for i, p := range params {
		if p.Grad == nil {
			analytic[i] = make([]float32, p.Data.Size())
			continue
		}
		analytic[i] = append([]float32(nil), p.Grad.Data()...)
	}

	for i, p := range params {
		d := p.Data.Data() // contiguous parameter: aliases the buffer
		for j := range d {
			orig := d[j]
			d[j] = orig + float32(eps)
			lp := float64(f().Item())
			d[j] = orig - float32(eps)
			lm := float64(f().Item())
			d[j] = orig

			num := (lp - lm) / (2 * eps)
			an := float64(analytic[i][j])
			rel := math.Abs(an-num) / math.Max(math.Abs(an)+math.Abs(num), 1e-3)
			if rel > maxRel {
				maxRel = rel
				where = fmt.Sprintf("param %d (%s) elem %d: analytic=%.6g numeric=%.6g", i, p.Name, j, an, num)
			}
		}
	}
	return maxRel, where
}
