package nn

import (
	"math"
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}

// rotated returns RoPE applied to vector v placed at absolute position pos.
func rotated(r *RoPE, v []float32, pos int) []float32 {
	x := autograd.NewVar(tensor.FromSlice(append([]float32(nil), v...), 1, len(v)))
	return r.Apply(x, pos).Data.Data()
}

// The key RoPE property (L6): ⟨R_m q, R_n k⟩ depends only on m − n.
func TestRoPERelativePosition(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	r := NewRoPE(8, 64, 10000)
	q := tensor.Randn(rng, 8).Data()
	k := tensor.Randn(rng, 8).Data()
	for _, mn := range [][2]int{{5, 2}, {9, 6}, {33, 30}, {3, 0}} { // m − n = 3 every time
		got := dot(rotated(r, q, mn[0]), rotated(r, k, mn[1]))
		want := dot(rotated(r, q, 3), rotated(r, k, 0))
		if math.Abs(got-want) > 1e-4 {
			t.Fatalf("m=%d n=%d: %.6f, want %.6f", mn[0], mn[1], got, want)
		}
	}
	// A rotation keeps the length of the vector.
	if a, b := dot(q, q), dot(rotated(r, q, 17), rotated(r, q, 17)); math.Abs(a-b) > 1e-4 {
		t.Fatalf("|q|² %.6f → %.6f", a, b)
	}
}

// The numeric example from L6: d = 2, one position = 30° (scale = 6/π turns angle p into p·π/6),
// q = (1, 0), k = (1, 1). Plain q·k = 1; with RoPE the score is cos 60° + sin 60° ≈ 1.3660 for any
// pair of positions with m − n = 2.
func TestRoPELectureExample(t *testing.T) {
	r := NewRoPEInterpolated(2, 8, 10000, 6/math.Pi) // with d = 2 the only frequency is ω₀ = 1
	q := rotated(r, []float32{1, 0}, 3)              // 90°: (0, 1)
	if math.Abs(float64(q[0])) > 1e-6 || math.Abs(float64(q[1])-1) > 1e-6 {
		t.Fatalf("R_3 (1,0) = %v, want (0, 1)", q)
	}
	k := rotated(r, []float32{1, 1}, 1) // 30°: (0.3660, 1.3660)
	if math.Abs(float64(k[0])-0.3660) > 1e-4 || math.Abs(float64(k[1])-1.3660) > 1e-4 {
		t.Fatalf("R_1 (1,1) = %v", k)
	}
	for _, mn := range [][2]int{{3, 1}, {5, 3}, {7, 5}} {
		got := dot(rotated(r, []float32{1, 0}, mn[0]), rotated(r, []float32{1, 1}, mn[1]))
		if math.Abs(got-1.3660) > 1e-4 {
			t.Fatalf("m=%d n=%d: score %.4f, want 1.3660", mn[0], mn[1], got)
		}
	}
}

// Sinusoidal PE: PE[p+k] is a fixed rotation of PE[p] (the same matrix for every p).
func TestSinusoidalShiftIsLinear(t *testing.T) {
	const dim, k = 8, 5
	pe := SinusoidalPE(64, dim)
	for _, p := range []int{0, 7, 40} {
		for i := range dim / 2 {
			w := math.Pow(10000, -float64(2*i)/dim)
			s, c := float64(pe.At(p, 2*i)), float64(pe.At(p, 2*i+1))
			// sin(a+b) = sin a cos b + cos a sin b;  cos(a+b) = cos a cos b − sin a sin b
			wantS := s*math.Cos(k*w) + c*math.Sin(k*w)
			wantC := c*math.Cos(k*w) - s*math.Sin(k*w)
			if math.Abs(wantS-float64(pe.At(p+k, 2*i))) > 1e-5 || math.Abs(wantC-float64(pe.At(p+k, 2*i+1))) > 1e-5 {
				t.Fatalf("p=%d i=%d: shift is not a rotation", p, i)
			}
		}
	}
	// Row 1, first pair: (sin 1, cos 1) = (0.8415, 0.5403).
	if math.Abs(float64(pe.At(1, 0))-0.8415) > 1e-4 || math.Abs(float64(pe.At(1, 1))-0.5403) > 1e-4 {
		t.Fatalf("PE[1] = %v %v", pe.At(1, 0), pe.At(1, 1))
	}
}

func TestCausalMask(t *testing.T) {
	inf := float32(math.Inf(-1))
	m := CausalMask(3, 3).Data()
	want := []float32{0, inf, inf, 0, 0, inf, 0, 0, 0}
	for i := range want {
		if m[i] != want[i] {
			t.Fatalf("mask(3,3) = %v", m)
		}
	}
	// One new query after 2 cached keys sees all 3 keys.
	for _, v := range CausalMask(1, 3).Data() {
		if v != 0 {
			t.Fatal("mask(1,3) must be all zeros")
		}
	}
}

func tinyAttention(rng *rand.Rand, heads, kvHeads int, rope bool) *CausalSelfAttention {
	var r *RoPE
	if rope {
		r = NewRoPE(8/heads, 16, 10000)
	}
	return NewCausalSelfAttention(rng, AttentionConfig{DModel: 8, Heads: heads, KVHeads: kvHeads, Bias: true, Std: 0.5, OutStd: 0.5}, r)
}

func TestAttentionShapesAndWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2))
	a := tinyAttention(rng, 4, 2, true)
	x := autograd.NewVar(tensor.Randn(rng, 2, 5, 8)) // (B, T, C)
	y := a.Forward(x, 0)
	if s := y.Shape(); s[0] != 2 || s[1] != 5 || s[2] != 8 {
		t.Fatalf("output shape %v", s)
	}
	w := a.Weights // (B, h, T, T)
	if s := w.Shape(); s[0] != 2 || s[1] != 4 || s[2] != 5 || s[3] != 5 {
		t.Fatalf("weights shape %v", s)
	}
	for bi := range 2 {
		for h := range 4 {
			for i := range 5 {
				var sum float64
				for j := range 5 {
					v := w.At(bi, h, i, j)
					if j > i && v != 0 {
						t.Fatalf("future weight [%d,%d,%d,%d] = %v", bi, h, i, j, v)
					}
					sum += float64(v)
				}
				if math.Abs(sum-1) > 1e-5 {
					t.Fatalf("row sum %v", sum)
				}
			}
		}
	}
}

// Causality: changing token 3 must not change outputs at positions 0..2.
func TestAttentionIsCausal(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	a := tinyAttention(rng, 2, 2, false)
	x := tensor.Randn(rng, 1, 4, 8)
	y1 := a.Forward(autograd.NewVar(x), 0).Data.Data()
	for c := range 8 {
		x.Set(x.At(0, 3, c)+10, 0, 3, c)
	}
	y2 := a.Forward(autograd.NewVar(x), 0).Data.Data()
	for i := range 3 * 8 {
		if y1[i] != y2[i] {
			t.Fatalf("output %d changed after editing a future token", i)
		}
	}
}

// Without positional information attention cannot tell order (L6): the last token
// sees every token, and its output is invariant to permuting the tokens before it.
func TestAttentionWithoutPositionsIgnoresOrder(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 4))
	a := tinyAttention(rng, 2, 2, false)
	x := tensor.Randn(rng, 1, 4, 8).Data()
	perm := []int{2, 0, 1, 3} // shuffle the first three, keep the last
	xp := make([]float32, len(x))
	for i, p := range perm {
		copy(xp[i*8:(i+1)*8], x[p*8:(p+1)*8])
	}
	last := func(d []float32) []float32 {
		return a.Forward(autograd.NewVar(tensor.FromSlice(d, 1, 4, 8)), 0).Data.Data()[3*8:]
	}
	y, yp := last(x), last(xp)
	for i := range y {
		if math.Abs(float64(y[i]-yp[i])) > 1e-5 {
			t.Fatalf("NoPE: last output changed under permutation: %v vs %v", y, yp)
		}
	}
	// With RoPE the same permutation changes the output.
	a.Rope = NewRoPE(4, 8, 10000)
	y, yp = last(x), last(xp)
	diff := 0.0
	for i := range y {
		diff = max(diff, math.Abs(float64(y[i]-yp[i])))
	}
	if diff < 1e-3 {
		t.Fatal("RoPE: output should depend on order")
	}
}

// GQA with h_kv = h must equal MHA; h_kv = 1 is MQA. All must pass gradcheck.
func TestAttentionGradCheck(t *testing.T) {
	for _, tc := range []struct {
		name      string
		heads, kv int
		rope      bool
	}{{"mha", 2, 2, false}, {"gqa-rope", 4, 2, true}, {"mqa-rope", 2, 1, true}} {
		rng := rand.New(rand.NewPCG(5, 5))
		a := tinyAttention(rng, tc.heads, tc.kv, tc.rope)
		x := autograd.Param(tensor.Randn(rng, 1, 3, 8))
		x.Name = "x"
		w := autograd.NewVar(tensor.Randn(rng, 1, 3, 8))
		loss := func() *autograd.Var { return autograd.Sum(autograd.Mul(a.Forward(x, 2), w)) }
		rel, where := autograd.GradCheck(loss, append(a.Parameters(), x), 1e-2)
		// 5e-2: the worst element is a W_K gradient of ~7e-4 whose finite difference
		// differs by ~4e-5 — float32 noise of a loss of order 1 at ε = 2e-2 (the fused
		// attention is compared with the composite one exactly in autograd's tests).
		if rel > 5e-2 {
			t.Errorf("%s: gradcheck %.3g at %s", tc.name, rel, where)
		}
	}
}

func TestBlockGradCheck(t *testing.T) {
	for _, tc := range []struct {
		name string
		cfg  BlockConfig
		rope bool
	}{
		{"gpt2", BlockConfig{Attention: AttentionConfig{DModel: 8, Heads: 2, Bias: true, Std: 0.5, OutStd: 0.5},
			FFHidden: 16, FFN: FFNGELU, Norm: NormLayer, NormEps: 1e-5}, false},
		{"llama", BlockConfig{Attention: AttentionConfig{DModel: 8, Heads: 2, KVHeads: 1, Std: 0.5, OutStd: 0.5},
			FFHidden: 12, FFN: FFNSwiGLU, Norm: NormRMS, NormEps: 1e-5}, true},
	} {
		rng := rand.New(rand.NewPCG(6, 6))
		var r *RoPE
		if tc.rope {
			r = NewRoPE(4, 8, 10000)
		}
		b := NewBlock(rng, tc.cfg, r)
		x := autograd.NewVar(tensor.Randn(rng, 2, 3, 8))
		w := autograd.NewVar(tensor.Randn(rng, 2, 3, 8))
		// Subtract the residual x: it adds a large constant to the loss and float32 noise with it.
		loss := func() *autograd.Var { return autograd.Sum(autograd.Mul(autograd.Sub(b.Forward(x, 0), x), w)) }
		// The key bias b_K gets an exactly zero gradient: it adds q·b_K to every score of a
		// row, and softmax ignores a constant shift (L7). Finite differences would only
		// measure float32 noise there, so it is checked separately.
		var ps []*autograd.Var
		for _, p := range b.Parameters() {
			if p != b.Attn.K.B {
				ps = append(ps, p)
			}
		}
		rel, where := autograd.GradCheck(loss, ps, 2e-2) // ε = 2e-2: tiny gradients of W_K drown in float32 noise at 1e-2
		// 5e-2: the worst element is a W_K gradient of ~7e-4 whose finite difference
		// differs by ~4e-5 — float32 noise of a loss of order 1 at ε = 2e-2 (the fused
		// attention is compared with the composite one exactly in autograd's tests).
		if rel > 5e-2 {
			t.Errorf("%s: gradcheck %.3g at %s", tc.name, rel, where)
		}
	}
}

func TestKeyBiasHasNoGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 7))
	a := tinyAttention(rng, 2, 2, false)
	x := autograd.NewVar(tensor.Randn(rng, 2, 4, 8))
	w := autograd.NewVar(tensor.Randn(rng, 2, 4, 8))
	autograd.Sum(autograd.Mul(a.Forward(x, 0), w)).Backward()
	for _, g := range a.K.B.Grad.Data() {
		if math.Abs(float64(g)) > 1e-5 {
			t.Fatalf("dL/db_K = %v, want 0", a.K.B.Grad.Data())
		}
	}
}

func TestEmbeddingForwardBatch(t *testing.T) {
	e := &Embedding{W: autograd.Param(tensor.FromSlice([]float32{1, 2, 3, 4, 5, 6}, 3, 2))}
	x := e.ForwardBatch([][]int{{2, 0}, {1, 1}})
	if s := x.Shape(); s[0] != 2 || s[1] != 2 || s[2] != 2 {
		t.Fatalf("shape %v", s)
	}
	if got := x.Data.Data(); got[0] != 5 || got[2] != 1 || got[4] != 3 || got[7] != 4 {
		t.Fatalf("values %v", got)
	}
}

// The two-token "it/works" example from L7, forward and backward, written with the
// same primitive ops as CausalSelfAttention (single head, d = 2, no projections).
func TestAttentionBackwardLectureExample(t *testing.T) {
	near := func(name string, got []float32, want ...float64) {
		t.Helper()
		for i := range want {
			if math.Abs(float64(got[i])-want[i]) > 1e-4 {
				t.Fatalf("%s = %v, want %v", name, got, want)
			}
		}
	}
	for _, causal := range []bool{false, true} {
		q := autograd.Param(tensor.FromSlice([]float32{0, 1, 2, 0}, 2, 2))
		k := autograd.Param(tensor.FromSlice([]float32{1, 0, 0, 2}, 2, 2))
		v := autograd.Param(tensor.FromSlice([]float32{2, 0, 0, 6}, 2, 2))
		s := autograd.Scale(autograd.MatMul(q, autograd.Transpose(k, 0, 1)), float32(1/math.Sqrt(2)))
		if causal {
			s = autograd.Add(s, autograd.NewVar(CausalMask(2, 2)))
		}
		a := autograd.Softmax(s)
		o := autograd.MatMul(a, v)
		dO := autograd.NewVar(tensor.FromSlice([]float32{1, 0, 0, 1}, 2, 2)) // L = o₁₁ + o₂₂
		autograd.Sum(autograd.Mul(o, dO)).Backward()
		if !causal {
			near("A", a.Data.Data(), 0.1956, 0.8044, 0.8044, 0.1956)
			near("O", o.Data.Data(), 0.3911, 4.8266, 1.6089, 1.1734)
			near("dV", v.Grad.Data(), 0.1956, 0.8044, 0.8044, 0.1956)
			near("dQ", q.Grad.Data(), 0.2225, -0.4450, -0.6675, 1.3349)
			near("dK", k.Grad.Data(), -1.3349, 0.2225, 1.3349, -0.2225)
		} else {
			near("A", a.Data.Data(), 1, 0, 0.8044, 0.1956)
			near("dQ", q.Grad.Data(), 0, 0, -0.6675, 1.3349)
			near("dK", k.Grad.Data(), -1.3349, 0, 1.3349, 0)
		}
	}
}
