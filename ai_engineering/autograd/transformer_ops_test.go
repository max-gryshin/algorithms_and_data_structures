package autograd

import (
	"math"
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// The LayerNorm/RMSNorm example from L8: x = [1, 2, 3, 6], g = 1, b = 0.
func TestNormLectureExample(t *testing.T) {
	x := Param(tensor.FromSlice([]float32{1, 2, 3, 6}, 1, 4))
	g := Param(tensor.FromSlice([]float32{1, 1, 1, 1}, 4))
	b := Param(tensor.New(4))
	// μ = 3, σ² = (4+1+0+9)/4 = 3.5, σ = 1.8708 → x̂ = [−1.0690, −0.5345, 0, 1.6036]
	ln := LayerNorm(x, g, b, 0)
	approx(t, ln.Data.Data(), []float32{-1.0690, -0.5345, 0, 1.6036}, 1e-4)
	// mean(x²) = (1+4+9+36)/4 = 12.5, r = 3.5355 → x̂ = [0.2828, 0.5657, 0.8485, 1.6971]
	rms := RMSNorm(x, g, 0)
	approx(t, rms.Data.Data(), []float32{0.2828, 0.5657, 0.8485, 1.6971}, 1e-4)
	// L = y[3] (only the last output): the gradient flows into every input because they share σ.
	// dx = (e₃ − mean(e₃) − x̂·x̂₃/4)/σ.
	Sum(Mul(ln, NewVar(tensor.FromSlice([]float32{0, 0, 0, 1}, 1, 4)))).Backward()
	approx(t, x.Grad.Data(), []float32{0.0955, -0.0191, -0.1336, 0.0573}, 1e-4)
}

// Fused ops must match the same formula composed from primitive ops.
func TestNormsMatchComposite(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	x := randParam(rng, "x", 2, 3, 5)
	g := randParam(rng, "g", 5)
	b := randParam(rng, "b", 5)
	const eps = 1e-5
	mu := MeanLast(x)
	xc := Sub(x, mu)
	compLN := Add(Mul(Div(xc, Pow(Add(MeanLast(Mul(xc, xc)), Const(eps)), 0.5)), g), b)
	if !tensor.AllClose(LayerNorm(x, g, b, eps).Data, compLN.Data, 1e-4, 1e-5) {
		t.Fatal("LayerNorm differs from composite")
	}
	compRMS := Mul(Mul(x, Pow(Add(MeanLast(Mul(x, x)), Const(eps)), -0.5)), g)
	if !tensor.AllClose(RMSNorm(x, g, eps).Data, compRMS.Data, 1e-4, 1e-5) {
		t.Fatal("RMSNorm differs from composite")
	}
}

func TestGradCheckTransformerOps(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	x := randParam(rng, "x", 2, 3, 6)
	g := randParam(rng, "g", 6)
	b := randParam(rng, "b", 6)
	w := NewVar(tensor.Randn(rng, 2, 3, 6)) // random weights make the loss sensitive to every output
	cos, sin := tensor.New(3, 3), tensor.New(3, 3)
	for p := range 3 {
		for i := range 3 {
			th := float64(p) * math.Pow(10, -float64(i))
			cos.Set(float32(math.Cos(th)), p, i)
			sin.Set(float32(math.Sin(th)), p, i)
		}
	}
	cases := map[string]struct {
		f      func() *Var
		params []*Var
	}{
		"layernorm":         {func() *Var { return Sum(Mul(LayerNorm(x, g, b, 1e-5), w)) }, []*Var{x, g, b}},
		"layernorm-no-bias": {func() *Var { return Sum(Mul(LayerNorm(x, g, nil, 1e-5), w)) }, []*Var{x, g}},
		"rmsnorm":           {func() *Var { return Sum(Mul(RMSNorm(x, g, 1e-5), w)) }, []*Var{x, g}},
		"rotary":            {func() *Var { return Sum(Mul(Tanh(Rotary(x, cos, sin)), w)) }, []*Var{x}},
	}
	for name, tc := range cases {
		rel, where := GradCheck(tc.f, tc.params, 1e-2)
		if rel > 2e-2 {
			t.Errorf("%s: max rel err %.3g at %s", name, rel, where)
		}
	}
}

// A 90° rotation of the pair (1, 0) gives (0, 1); position 0 (θ = 0) is untouched.
func TestRotaryRotatesPairs(t *testing.T) {
	x := NewVar(tensor.FromSlice([]float32{1, 0, 1, 0}, 2, 2)) // T = 2 rows, d = 2
	cos := tensor.FromSlice([]float32{1, 0}, 2, 1)             // θ = 0, π/2
	sin := tensor.FromSlice([]float32{0, 1}, 2, 1)
	approx(t, Rotary(x, cos, sin).Data.Data(), []float32{1, 0, 0, 1}, 1e-7)
}
