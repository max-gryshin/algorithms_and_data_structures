// Package optim implements gradient-based optimizers over autograd parameters
// (lecture L3): SGD, Momentum, RMSProp, Adam, AdamW, plus a warmup+cosine
// learning-rate schedule and global gradient-norm clipping.
//
// Typical training step:
//
//	opt.ZeroGrad()
//	loss := model.Loss(batch)
//	loss.Backward()
//	optim.ClipGradNorm(params, 1.0)
//	opt.SetLR(sched.LR(step))
//	opt.Step()
package optim

import (
	"fmt"
	"math"

	"algorithms_and_data_structures/ai_engineering/autograd"
)

// Optimizer updates parameters in place from their accumulated gradients.
type Optimizer interface {
	Step()
	ZeroGrad()
	SetLR(lr float64)
	LR() float64
}

type base struct {
	params []*autograd.Var
	lr     float64
}

func (b *base) ZeroGrad() {
	for _, p := range b.params {
		p.ZeroGrad()
	}
}
func (b *base) SetLR(lr float64) { b.lr = lr }
func (b *base) LR() float64      { return b.lr }

// each calls f(i, w, g) for every element of every parameter that has a gradient;
// state slices are indexed by parameter k and element i.
func (b *base) each(f func(k, i int, w []float32, g []float32)) {
	for k, p := range b.params {
		if p.Grad == nil {
			continue
		}
		w, g := p.Data.Data(), p.Grad.Data()
		for i := range w {
			f(k, i, w, g)
		}
	}
}

func (b *base) zeros() [][]float64 {
	s := make([][]float64, len(b.params))
	for k, p := range b.params {
		s[k] = make([]float64, p.Data.Size())
	}
	return s
}

// SGD: w ← w − lr·g.
type SGD struct{ base }

func NewSGD(params []*autograd.Var, lr float64) *SGD {
	return &SGD{base{params: params, lr: lr}}
}

func (o *SGD) Step() {
	o.each(func(_, i int, w, g []float32) {
		w[i] -= float32(o.lr * float64(g[i]))
	})
}

// Momentum (heavy ball, PyTorch convention):
//
//	v ← β·v + g
//	w ← w − lr·v
type Momentum struct {
	base
	Beta float64
	v    [][]float64
}

func NewMomentum(params []*autograd.Var, lr, beta float64) *Momentum {
	o := &Momentum{base: base{params: params, lr: lr}, Beta: beta}
	o.v = o.zeros()
	return o
}

func (o *Momentum) Step() {
	o.each(func(k, i int, w, g []float32) {
		v := o.Beta*o.v[k][i] + float64(g[i])
		o.v[k][i] = v
		w[i] -= float32(o.lr * v)
	})
}

// RMSProp divides the step by a running RMS of the gradient:
//
//	s ← β·s + (1−β)·g²
//	w ← w − lr·g / (√s + ε)
type RMSProp struct {
	base
	Beta, Eps float64
	s         [][]float64
}

func NewRMSProp(params []*autograd.Var, lr float64) *RMSProp {
	o := &RMSProp{base: base{params: params, lr: lr}, Beta: 0.99, Eps: 1e-8}
	o.s = o.zeros()
	return o
}

func (o *RMSProp) Step() {
	o.each(func(k, i int, w, g []float32) {
		gi := float64(g[i])
		s := o.Beta*o.s[k][i] + (1-o.Beta)*gi*gi
		o.s[k][i] = s
		w[i] -= float32(o.lr * gi / (math.Sqrt(s) + o.Eps))
	})
}

// Adam keeps running averages of the gradient (m) and of its square (v):
//
//	m ← β₁·m + (1−β₁)·g            v ← β₂·v + (1−β₂)·g²
//	m̂ = m / (1−β₁ᵗ)                v̂ = v / (1−β₂ᵗ)          (bias correction)
//	w ← w − lr·m̂ / (√v̂ + ε)
//
// Weight decay is applied in one of two ways:
//
//	Decoupled = false (classic Adam + L2): g ← g + λ·w before the moments;
//	Decoupled = true  (AdamW):            w ← w − lr·λ·w, separately from the moments.
//
// AdamW decays only parameters with ≥ 2 dimensions (weight matrices and embeddings),
// not biases and norm gains — the nanoGPT/GPT-2 convention.
type Adam struct {
	base
	Beta1, Beta2, Eps float64
	WeightDecay       float64
	Decoupled         bool

	t    int
	m, v [][]float64
}

// NewAdam returns Adam with the usual defaults β₁=0.9, β₂=0.999, ε=1e-8 and no weight decay.
func NewAdam(params []*autograd.Var, lr float64) *Adam {
	o := &Adam{base: base{params: params, lr: lr}, Beta1: 0.9, Beta2: 0.999, Eps: 1e-8}
	o.m, o.v = o.zeros(), o.zeros()
	return o
}

// NewAdamW returns AdamW with β₁=0.9, β₂=0.95 (the LLM-training default) and decoupled weight decay.
func NewAdamW(params []*autograd.Var, lr, weightDecay float64) *Adam {
	o := NewAdam(params, lr)
	o.Beta2 = 0.95
	o.WeightDecay = weightDecay
	o.Decoupled = true
	return o
}

func (o *Adam) Step() {
	o.t++
	bc1 := 1 - math.Pow(o.Beta1, float64(o.t))
	bc2 := 1 - math.Pow(o.Beta2, float64(o.t))
	for k, p := range o.params {
		if p.Grad == nil {
			continue
		}
		decay := o.WeightDecay
		if o.Decoupled && p.Data.Dims() < 2 {
			decay = 0
		}
		w, g := p.Data.Data(), p.Grad.Data()
		m, v := o.m[k], o.v[k]
		for i := range w {
			wi, gi := float64(w[i]), float64(g[i])
			if !o.Decoupled {
				gi += decay * wi // L2: the decay goes through the adaptive scaling
			}
			m[i] = o.Beta1*m[i] + (1-o.Beta1)*gi
			v[i] = o.Beta2*v[i] + (1-o.Beta2)*gi*gi
			mHat, vHat := m[i]/bc1, v[i]/bc2
			if o.Decoupled {
				wi -= o.lr * decay * wi // AdamW: plain shrink, independent of the gradient history
			}
			w[i] = float32(wi - o.lr*mHat/(math.Sqrt(vHat)+o.Eps))
		}
	}
}

// CosineSchedule: linear warmup from 0 to MaxLR over WarmupSteps, then cosine decay
// to MinLR at TotalSteps, constant MinLR afterwards.
//
//	step < W:      lr = MaxLR · (step+1)/W
//	W ≤ step ≤ T:  lr = MinLR + ½(MaxLR − MinLR)(1 + cos(π·(step−W)/(T−W)))
type CosineSchedule struct {
	MaxLR, MinLR float64
	WarmupSteps  int
	TotalSteps   int
}

func (s CosineSchedule) LR(step int) float64 {
	if step < s.WarmupSteps {
		return s.MaxLR * float64(step+1) / float64(s.WarmupSteps)
	}
	if step >= s.TotalSteps {
		return s.MinLR
	}
	progress := float64(step-s.WarmupSteps) / float64(s.TotalSteps-s.WarmupSteps)
	return s.MinLR + 0.5*(s.MaxLR-s.MinLR)*(1+math.Cos(math.Pi*progress))
}

// GradNorm returns the global L2 norm of all gradients: √(Σ over all params and elements g²).
func GradNorm(params []*autograd.Var) float64 {
	var s float64
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		for _, g := range p.Grad.Data() {
			s += float64(g) * float64(g)
		}
	}
	return math.Sqrt(s)
}

// ClipGradNorm rescales all gradients together so that their global norm is at most maxNorm.
// The direction of the full gradient is preserved. It returns the norm before clipping
// (worth logging: spikes are an early sign of instability).
func ClipGradNorm(params []*autograd.Var, maxNorm float64) float64 {
	norm := GradNorm(params)
	if norm <= maxNorm || norm == 0 {
		return norm
	}
	scale := float32(maxNorm / norm)
	for _, p := range params {
		if p.Grad == nil {
			continue
		}
		g := p.Grad.Data()
		for i := range g {
			g[i] *= scale
		}
	}
	return norm
}

// State exposes Adam's internal state for checkpoints (L9): the number of steps taken
// (for bias correction) and the moments m, v per parameter (aliased, not copied).
// Without them a resumed run restarts with m = v = 0 — effectively a fresh warmup.
func (o *Adam) State() (t int, m, v [][]float64) { return o.t, o.m, o.v }

// SetState restores what State returned; shapes must match the parameters.
func (o *Adam) SetState(t int, m, v [][]float64) error {
	if len(m) != len(o.params) || len(v) != len(o.params) {
		return fmt.Errorf("optim: state for %d/%d params, have %d", len(m), len(v), len(o.params))
	}
	for k, p := range o.params {
		if len(m[k]) != p.Data.Size() || len(v[k]) != p.Data.Size() {
			return fmt.Errorf("optim: state of param %d has %d/%d values, want %d", k, len(m[k]), len(v[k]), p.Data.Size())
		}
	}
	o.t, o.m, o.v = t, m, v
	return nil
}
