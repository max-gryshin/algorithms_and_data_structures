package autograd

import (
	"math"
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

func randParam(rng *rand.Rand, name string, shape ...int) *Var {
	p := Param(tensor.Randn(rng, shape...))
	p.Name = name
	return p
}

func approx(t *testing.T, got, want []float32, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("len %d, want %d", len(got), len(want))
	}
	for i := range got {
		if math.Abs(float64(got[i]-want[i])) > tol {
			t.Fatalf("got %v, want %v", got, want)
		}
	}
}

// The 2×2 example from L3: L = sum(A·B)  =>  dA = 1·Bᵀ, dB = Aᵀ·1.
func TestMatMulGradLectureExample(t *testing.T) {
	a := Param(tensor.FromSlice([]float32{1, 2, 3, 4}, 2, 2))
	b := Param(tensor.FromSlice([]float32{5, 6, 7, 8}, 2, 2))
	Sum(MatMul(a, b)).Backward()
	approx(t, a.Grad.Data(), []float32{11, 15, 11, 15}, 0)
	approx(t, b.Grad.Data(), []float32{4, 4, 6, 6}, 0)
}

// The softmax + cross-entropy example from L3: gradient = p − y.
func TestCrossEntropyLectureExample(t *testing.T) {
	z := Param(tensor.FromSlice([]float32{2, 1, 0.1}, 1, 3))
	loss := CrossEntropy(z, []int{0})
	loss.Backward()
	approx(t, []float32{loss.Item()}, []float32{0.4170}, 1e-4)
	approx(t, z.Grad.Data(), []float32{0.6590 - 1, 0.2424, 0.0986}, 1e-4)
}

func TestCrossEntropyIgnoresNegativeTargets(t *testing.T) {
	z := Param(tensor.FromSlice([]float32{2, 1, 0.1, 5, 5, 5}, 2, 3))
	loss := CrossEntropy(z, []int{0, -1})
	loss.Backward()
	approx(t, []float32{loss.Item()}, []float32{0.4170}, 1e-4)
	approx(t, z.Grad.Data()[3:], []float32{0, 0, 0}, 0)
}

func TestBroadcastAddReducesGrad(t *testing.T) {
	x := Param(tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 2, 3))
	bias := Param(tensor.FromSlice([]float32{0, 0, 0}, 3))
	Sum(Add(x, bias)).Backward()
	approx(t, bias.Grad.Data(), []float32{2, 2, 2}, 0) // each bias element was used in 2 rows
}

func TestGradAccumulatesAcrossBackward(t *testing.T) {
	w := Param(tensor.FromSlice([]float32{3}))
	Mul(w, w).Backward()
	Mul(w, w).Backward()
	approx(t, w.Grad.Data(), []float32{12}, 0)
}

func TestEmbeddingScatterAdd(t *testing.T) {
	w := Param(tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 3, 2))
	e := Embedding(w, []int{2, 0, 2})
	approx(t, e.Data.Data(), []float32{5, 6, 1, 2, 5, 6}, 0)
	Sum(e).Backward()
	approx(t, w.Grad.Data(), []float32{1, 1, 0, 0, 2, 2}, 0) // id 2 used twice
}

// Every tensor op is verified against finite differences.
func TestGradCheckOps(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 1))
	a := randParam(rng, "a", 2, 3, 4)
	b := randParam(rng, "b", 4, 5)
	c := randParam(rng, "c", 4)
	pos := Param(tensor.Randn(rng, 2, 3).Map(func(x float32) float32 { return 1 + x*x }))
	pos.Name = "pos"
	targets := []int{0, 4, 2, 1, 3, 4}

	cases := map[string]struct {
		f      func() *Var
		params []*Var
	}{
		"matmul-broadcast": {func() *Var { return Sum(Tanh(MatMul(a, b))) }, []*Var{a, b}},
		"add-broadcast":    {func() *Var { return Sum(Mul(Add(a, c), Add(a, c))) }, []*Var{a, c}},
		"sub-mul-div":      {func() *Var { return Mean(Div(Sub(a, c), Add(Mul(c, c), Const(1)))) }, []*Var{a, c}},
		"pow-exp-log":      {func() *Var { return Sum(Log(Add(Exp(Pow(pos, 0.5)), Const(1)))) }, []*Var{pos}},
		"activations": {func() *Var {
			return Sum(Mul(Add(Add(GELU(a), SiLU(a)), Sigmoid(a)), Tanh(a)))
		}, []*Var{a}},
		"softmax":         {func() *Var { return Sum(Mul(Softmax(a), a)) }, []*Var{a}},
		"sumlast-meanlst": {func() *Var { return Sum(Mul(SumLast(a), MeanLast(Mul(a, a)))) }, []*Var{a}},
		"reshape-transpose": {func() *Var {
			return Sum(Mul(Transpose(Reshape(a, 6, 4), 0, 1), Reshape(Transpose(a, 1, 2), 4, 6)))
		}, []*Var{a}},
		"embedding-ce": {func() *Var {
			return CrossEntropy(MatMul(Embedding(a.reshapeHack(), []int{0, 3, 3, 5, 1, 0}), b), targets)
		}, []*Var{a, b}},
	}
	for name, tc := range cases {
		rel, where := GradCheck(tc.f, tc.params, 1e-2)
		if rel > 2e-2 {
			t.Errorf("%s: max rel err %.3g at %s", name, rel, where)
		}
	}
}

// reshapeHack views the (2,3,4) parameter as a (6,4) embedding table — exercising
// gradient flow through Reshape into Embedding.
func (v *Var) reshapeHack() *Var { return Reshape(v, 6, 4) }

func TestReLUGrad(t *testing.T) {
	x := Param(tensor.FromSlice([]float32{-1, 0.5, 2}, 3))
	Sum(ReLU(x)).Backward()
	approx(t, x.Grad.Data(), []float32{0, 1, 1}, 0)
}

func TestConstantsGetNoGrad(t *testing.T) {
	c := NewVar(tensor.FromSlice([]float32{1, 2}, 2))
	w := Param(tensor.FromSlice([]float32{3, 4}, 2))
	Sum(Mul(c, w)).Backward()
	if c.Grad != nil {
		t.Fatal("constant received a gradient")
	}
	approx(t, w.Grad.Data(), []float32{1, 2}, 0)
}
