package optim

import (
	"math"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// The running example of L3: f(w) = (w − 3)², w₀ = 0, f'(w) = 2(w − 3).
func runQuadratic(t *testing.T, newOpt func(p []*autograd.Var) Optimizer, want []float64) {
	t.Helper()
	w := autograd.Param(tensor.FromSlice([]float32{0}, 1, 1))
	opt := newOpt([]*autograd.Var{w})
	for step, wt := range want {
		opt.ZeroGrad()
		d := autograd.Sub(w, autograd.Const(3))
		autograd.Sum(autograd.Mul(d, d)).Backward()
		opt.Step()
		if got := float64(w.Data.Data()[0]); math.Abs(got-wt) > 1e-5 {
			t.Fatalf("step %d: w = %.6f, want %.6f", step+1, got, wt)
		}
	}
}

func TestSGD(t *testing.T) {
	runQuadratic(t, func(p []*autograd.Var) Optimizer { return NewSGD(p, 0.1) }, []float64{0.6, 1.08, 1.464})
}

func TestMomentum(t *testing.T) {
	runQuadratic(t, func(p []*autograd.Var) Optimizer { return NewMomentum(p, 0.1, 0.9) }, []float64{0.6, 1.62, 2.814})
}

func TestRMSProp(t *testing.T) {
	runQuadratic(t, func(p []*autograd.Var) Optimizer {
		o := NewRMSProp(p, 0.1)
		o.Beta = 0.9
		return o
	}, []float64{0.316228, 0.533179, 0.708234})
}

func TestAdam(t *testing.T) {
	runQuadratic(t, func(p []*autograd.Var) Optimizer { return NewAdam(p, 0.1) }, []float64{0.1, 0.199897, 0.299618})
}

// Adam's step does not depend on the gradient scale: f = 100·(w−3)² gives the same trajectory.
func TestAdamScaleInvariant(t *testing.T) {
	w := autograd.Param(tensor.FromSlice([]float32{0}, 1, 1))
	opt := NewAdam([]*autograd.Var{w}, 0.1)
	for _, want := range []float64{0.1, 0.199897, 0.299618} {
		opt.ZeroGrad()
		d := autograd.Sub(w, autograd.Const(3))
		autograd.Scale(autograd.Sum(autograd.Mul(d, d)), 100).Backward()
		opt.Step()
		if got := float64(w.Data.Data()[0]); math.Abs(got-want) > 1e-5 {
			t.Fatalf("w = %.6f, want %.6f", got, want)
		}
	}
}

// L2-in-Adam vs decoupled AdamW, one step from w = 1 with λ = 0.1, lr = 0.1,
// for a parameter with loss gradient 0 and one with loss gradient 10.
func TestAdamL2VersusAdamW(t *testing.T) {
	cases := []struct {
		decoupled bool
		grad      float32
		want      float64
	}{
		{false, 0, 0.9}, {false, 10, 0.9}, // L2: decay is either a full lr step or invisible
		{true, 0, 0.99}, {true, 10, 0.89}, // AdamW: always exactly lr·λ·w = 0.01
	}
	for _, c := range cases {
		w := autograd.Param(tensor.FromSlice([]float32{1}, 1, 1))
		opt := NewAdam([]*autograd.Var{w}, 0.1)
		opt.WeightDecay, opt.Decoupled = 0.1, c.decoupled
		w.Grad = tensor.FromSlice([]float32{c.grad}, 1, 1)
		opt.Step()
		if got := float64(w.Data.Data()[0]); math.Abs(got-c.want) > 1e-6 {
			t.Errorf("decoupled=%v grad=%v: w = %.6f, want %.6f", c.decoupled, c.grad, got, c.want)
		}
	}
}

func TestAdamWSkipsVectors(t *testing.T) {
	bias := autograd.Param(tensor.FromSlice([]float32{1}, 1))
	opt := NewAdamW([]*autograd.Var{bias}, 0.1, 0.5)
	bias.Grad = tensor.FromSlice([]float32{0}, 1)
	opt.Step()
	if bias.Data.Data()[0] != 1 {
		t.Fatalf("bias was decayed: %v", bias.Data.Data()[0])
	}
}

func TestCosineSchedule(t *testing.T) {
	s := CosineSchedule{MaxLR: 1e-3, MinLR: 1e-4, WarmupSteps: 100, TotalSteps: 1000}
	for step, want := range map[int]float64{0: 1e-5, 49: 5e-4, 99: 1e-3, 100: 1e-3, 550: 5.5e-4, 1000: 1e-4, 5000: 1e-4} {
		if got := s.LR(step); math.Abs(got-want) > 1e-12 {
			t.Errorf("LR(%d) = %g, want %g", step, got, want)
		}
	}
}

func TestClipGradNorm(t *testing.T) {
	a := autograd.Param(tensor.New(2))
	b := autograd.Param(tensor.New(1))
	a.Grad = tensor.FromSlice([]float32{3, 4}, 2)
	b.Grad = tensor.FromSlice([]float32{12}, 1)
	if n := ClipGradNorm([]*autograd.Var{a, b}, 1); n != 13 {
		t.Fatalf("norm = %v, want 13", n)
	}
	if n := GradNorm([]*autograd.Var{a, b}); math.Abs(n-1) > 1e-6 {
		t.Fatalf("norm after clip = %v, want 1", n)
	}
	if g := a.Grad.Data(); math.Abs(float64(g[0])-3.0/13) > 1e-6 {
		t.Fatalf("direction changed: %v", g)
	}
}
