package autograd

import (
	"math"
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// compositeAttention is the unfused reference built from primitive ops, with the
// bottom-right causal mask and K/V heads repeated for GQA.
func compositeAttention(q, k, v *Var, scale float32) *Var {
	qs, ks := q.Shape(), k.Shape()
	b, h, tq, d := qs[0], qs[1], qs[2], qs[3]
	kvh, tk := ks[1], ks[2]
	g := h / kvh
	mask := tensor.New(tq, tk)
	for i := range tq {
		for j := tk - tq + i + 1; j < tk; j++ {
			mask.Set(float32(math.Inf(-1)), i, j)
		}
	}
	q5 := Reshape(q, b, kvh, g, tq, d)
	k5 := Reshape(k, b, kvh, 1, tk, d)
	v5 := Reshape(v, b, kvh, 1, tk, d)
	s := Scale(MatMul(q5, Transpose(k5, 3, 4)), scale)
	p := Softmax(Add(s, NewVar(mask)))
	return Reshape(MatMul(p, v5), b, h, tq, d)
}

func TestCausalAttentionMatchesComposite(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	for _, c := range []struct{ b, h, kv, tq, tk, d int }{
		{2, 4, 4, 5, 5, 3}, // MHA, full sequence
		{1, 4, 2, 3, 3, 4}, // GQA
		{2, 2, 1, 2, 6, 2}, // MQA, 2 new queries against 6 keys (KV-cache shape)
	} {
		q := randParam(rng, "q", c.b, c.h, c.tq, c.d)
		k := randParam(rng, "k", c.b, c.kv, c.tk, c.d)
		v := randParam(rng, "v", c.b, c.kv, c.tk, c.d)
		w := NewVar(tensor.Randn(rng, c.b, c.h, c.tq, c.d))
		scale := float32(1 / math.Sqrt(float64(c.d)))

		fused, _ := CausalAttention(q, k, v, scale)
		ref := compositeAttention(q, k, v, scale)
		if !tensor.AllClose(fused.Data, ref.Data, 1e-5, 1e-6) {
			t.Fatalf("%+v: forward differs", c)
		}
		Sum(Mul(fused, w)).Backward()
		grads := []*tensor.Tensor{q.Grad, k.Grad, v.Grad}
		for _, p := range []*Var{q, k, v} {
			p.ZeroGrad()
		}
		Sum(Mul(ref, w)).Backward()
		for i, p := range []*Var{q, k, v} {
			if !tensor.AllClose(grads[i], p.Grad, 1e-4, 1e-5) {
				t.Fatalf("%+v: grad %d differs:\n%v\n%v", c, i, grads[i], p.Grad)
			}
		}
		rel, where := GradCheck(func() *Var {
			o, _ := CausalAttention(q, k, v, scale)
			return Sum(Mul(Tanh(o), w))
		}, []*Var{q, k, v}, 1e-2)
		if rel > 5e-2 { // float32 finite differences; the exact check is the comparison above
			t.Errorf("%+v: gradcheck rel err %.3g at %s", c, rel, where)
		}
	}
}

// The L7 "it/works" example: causal, one head, d = 2.
func TestCausalAttentionLectureExample(t *testing.T) {
	q := Param(tensor.FromSlice([]float32{0, 1, 2, 0}, 1, 1, 2, 2))
	k := Param(tensor.FromSlice([]float32{1, 0, 0, 2}, 1, 1, 2, 2))
	v := Param(tensor.FromSlice([]float32{2, 0, 0, 6}, 1, 1, 2, 2))
	o, p := CausalAttention(q, k, v, float32(1/math.Sqrt(2)))
	approx(t, p.Data(), []float32{1, 0, 0.8044, 0.1956}, 1e-4)
	Sum(Mul(o, NewVar(tensor.FromSlice([]float32{1, 0, 0, 1}, 1, 1, 2, 2)))).Backward()
	approx(t, q.Grad.Data(), []float32{0, 0, -0.6675, 1.3349}, 1e-4)
	approx(t, k.Grad.Data(), []float32{-1.3349, 0, 1.3349, 0}, 1e-4)
}

func TestMatMulFastPathsGradCheck(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	x := randParam(rng, "x", 2, 3, 4)
	w := randParam(rng, "w", 4, 5)
	e := randParam(rng, "e", 5, 4) // (V, d) for the tied head x·Eᵀ
	bias := randParam(rng, "b", 5)
	r := NewVar(tensor.Randn(rng, 2, 3, 5))
	r4 := NewVar(tensor.Randn(rng, 2, 3, 4))
	cases := map[string]struct {
		f      func() *Var
		params []*Var
	}{
		"linear":   {func() *Var { return Sum(Mul(Tanh(Add(MatMul(x, w), bias)), r)) }, []*Var{x, w, bias}},
		"tiedhead": {func() *Var { return Sum(Mul(Tanh(MatMulBT(x, e)), r)) }, []*Var{x, e}},
		"transpose": {func() *Var {
			return Sum(Mul(Tanh(Reshape(Transpose(Reshape(x, 2, 3, 2, 2), 1, 2), 2, 3, 4)), r4))
		}, []*Var{x}},
	}
	for name, tc := range cases {
		if rel, where := GradCheck(tc.f, tc.params, 1e-2); rel > 2e-2 {
			t.Errorf("%s: gradcheck rel err %.3g at %s", name, rel, where)
		}
	}
	// MatMulBT equals MatMul with an explicit transpose; results do not depend on threads.
	want := MatMul(x, Transpose(e, 0, 1)).Data
	old := SetThreads(1)
	got1 := MatMulBT(x, e).Data
	SetThreads(old)
	if !tensor.AllClose(want, got1, 1e-6, 1e-6) || !tensor.AllClose(MatMulBT(x, e).Data, got1, 0, 0) {
		t.Fatal("MatMulBT differs")
	}
}
