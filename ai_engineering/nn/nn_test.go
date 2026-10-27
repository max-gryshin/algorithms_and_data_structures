package nn

import (
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

func TestLinearShapesAndGrad(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	l := NewLinear(rng, 4, 3, true, 1)
	x := autograd.NewVar(tensor.Randn(rng, 2, 5, 4)) // (B, T, In)
	y := l.Forward(x)
	if s := y.Shape(); s[0] != 2 || s[1] != 5 || s[2] != 3 {
		t.Fatalf("shape = %v, want [2 5 3]", s)
	}
	rel, where := autograd.GradCheck(func() *autograd.Var {
		return autograd.Mean(autograd.Tanh(l.Forward(x)))
	}, l.Parameters(), 1e-2)
	if rel > 2e-2 {
		t.Fatalf("gradcheck: %.3g at %s", rel, where)
	}
}

func TestMLPGradCheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	emb := NewEmbedding(rng, 7, 3, 1)
	mlp := NewMLP(rng, []int{6, 8, 7}, autograd.GELU, 1)
	ids := []int{1, 2, 3, 4, 5, 6, 0, 0}
	targets := []int{3, 0, 6, 2}
	loss := func() *autograd.Var {
		x := autograd.Reshape(emb.Forward(ids), 4, 6) // 4 examples × (2 tokens · 3 dims)
		return autograd.CrossEntropy(mlp.Forward(x), targets)
	}
	ps := Collect(emb, mlp)
	if n := CountParams(ps); n != 7*3+6*8+8+8*7+7 {
		t.Fatalf("CountParams = %d", n)
	}
	rel, where := autograd.GradCheck(loss, ps, 1e-2)
	if rel > 2e-2 {
		t.Fatalf("gradcheck: %.3g at %s", rel, where)
	}
}
