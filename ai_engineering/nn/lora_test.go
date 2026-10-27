package nn

import (
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// At init B = 0: the adapted layer is exactly the base layer.
func TestLoRAStartsAsIdentityUpdate(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	l := NewLinear(rng, 6, 5, true, 1)
	x := autograd.NewVar(tensor.Randn(rng, 3, 6))
	before := l.Forward(x).Data
	l.AddLoRA(rng, 2, 4)
	if !tensor.AllClose(before, l.Forward(x).Data, 0, 0) {
		t.Fatal("LoRA with B = 0 changed the output")
	}
}

func TestLoRAGradCheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2))
	l := NewLinear(rng, 4, 3, true, 1)
	a := l.AddLoRA(rng, 2, 4)
	a.B = Normal(rng, "lora.B", 0.5, 2, 3) // B = 0 would make dA = 0 — check a generic point
	x := autograd.NewVar(tensor.Randn(rng, 2, 5, 4))
	f := func() *autograd.Var { return autograd.Mean(autograd.Tanh(l.Forward(x))) }
	rel, where := autograd.GradCheck(f, append(a.Parameters(), l.Parameters()...), 1e-2)
	if rel > 2e-2 {
		t.Fatalf("gradcheck: %.3g at %s", rel, where)
	}
}

// With B = 0 the gradient reaches B but not A — the reason for the asymmetric init.
func TestLoRAFirstGradientOnlyInB(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	l := NewLinear(rng, 4, 3, false, 1)
	a := l.AddLoRA(rng, 2, 2)
	l.W.RequiresGrad = false // frozen base
	x := autograd.NewVar(tensor.Randn(rng, 5, 4))
	autograd.Mean(autograd.Tanh(l.Forward(x))).Backward()
	if l.W.Grad != nil {
		t.Fatal("frozen W got a gradient")
	}
	if tensor.AllClose(a.B.Grad, tensor.New(2, 3), 0, 1e-12) {
		t.Fatal("dB is zero")
	}
	if !tensor.AllClose(a.A.Grad, tensor.New(4, 2), 0, 1e-12) {
		t.Fatal("dA must be zero while B = 0")
	}
}

func TestLoRAMergeUnmerge(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	l := NewLinear(rng, 8, 6, true, 1)
	w0 := tensor.FromSlice(append([]float32(nil), l.W.Data.Data()...), 8, 6)
	a := l.AddLoRA(rng, 3, 6)
	a.B = Normal(rng, "lora.B", 0.3, 3, 6)
	x := autograd.NewVar(tensor.Randn(rng, 2, 4, 8))

	adapted := l.Forward(x).Data
	l.Merge()
	if !tensor.AllClose(adapted, l.Forward(x).Data, 1e-5, 1e-5) {
		t.Fatal("merged layer differs from adapter branch")
	}
	l.Merge() // idempotent
	l.Unmerge()
	if !tensor.AllClose(w0, l.W.Data, 1e-6, 1e-6) {
		t.Fatal("unmerge did not restore W")
	}
	if !tensor.AllClose(adapted, l.Forward(x).Data, 1e-5, 1e-5) {
		t.Fatal("unmerged layer differs")
	}
	if n := CountParams(a.Parameters()); n != 3*(8+6) {
		t.Fatalf("adapter params %d, want r·(in+out) = 42", n)
	}
}
