// Package nn holds neural-network building blocks on top of autograd (lecture L4):
// Linear, Embedding, MLP. Later lectures add attention, norms and the transformer block.
//
// Conventions:
//   - activations are (..., features): the last axis is the feature axis,
//     every leading axis (batch, time, ...) is broadcast;
//   - a Linear layer computes x·W + b with W of shape (in, out), so rows of x
//     are tokens/examples and the matmul needs no transpose;
//   - every module exposes its trainable tensors via Parameters().
package nn

import (
	"math"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Module is anything with trainable parameters.
type Module interface {
	Parameters() []*autograd.Var
}

// Collect concatenates the parameters of several modules.
func Collect(mods ...Module) []*autograd.Var {
	var ps []*autograd.Var
	for _, m := range mods {
		ps = append(ps, m.Parameters()...)
	}
	return ps
}

// CountParams returns the total number of scalar parameters.
func CountParams(ps []*autograd.Var) int {
	n := 0
	for _, p := range ps {
		n += p.Data.Size()
	}
	return n
}

// Normal returns a parameter with elements drawn from N(0, std²).
func Normal(rng *rand.Rand, name string, std float32, shape ...int) *autograd.Var {
	p := autograd.Param(tensor.Randn(rng, shape...).Scale(std))
	p.Name = name
	return p
}

// Zeros returns a zero-initialised parameter.
func Zeros(name string, shape ...int) *autograd.Var {
	p := autograd.Param(tensor.New(shape...))
	p.Name = name
	return p
}

// Linear: y = x·W + b, x: (..., In), W: (In, Out), b: (Out) -> y: (..., Out).
type Linear struct {
	W, B *autograd.Var // B is nil when the layer has no bias

	// LoRA is an optional low-rank adapter (L13): when set and not merged, Forward adds
	// (α/r)·(x·A)·B. It is not part of Parameters() — the adapter is trained separately
	// (LoRA.Parameters), the frozen base weights keep their names and checkpoints.
	LoRA *LoRA
}

// NewLinear initialises W ~ N(0, gain²/In) (gain = 1 for linear/tanh-like outputs,
// √2 for ReLU — see He init in L4) and b = 0.
func NewLinear(rng *rand.Rand, in, out int, bias bool, gain float64) *Linear {
	l := &Linear{W: Normal(rng, "linear.W", float32(gain/math.Sqrt(float64(in))), in, out)}
	if bias {
		l.B = Zeros("linear.b", out)
	}
	return l
}

func (l *Linear) Forward(x *autograd.Var) *autograd.Var {
	y := autograd.MatMul(x, l.W)
	if l.B != nil {
		y = autograd.Add(y, l.B)
	}
	if l.LoRA != nil && !l.LoRA.Merged {
		y = autograd.Add(y, l.LoRA.Forward(x))
	}
	return y
}

func (l *Linear) Parameters() []*autograd.Var {
	if l.B == nil {
		return []*autograd.Var{l.W}
	}
	return []*autograd.Var{l.W, l.B}
}

// Embedding is a lookup table W: (Num, Dim); Forward(ids) returns (len(ids), Dim).
type Embedding struct {
	W *autograd.Var
}

func NewEmbedding(rng *rand.Rand, num, dim int, std float32) *Embedding {
	return &Embedding{W: Normal(rng, "embedding.W", std, num, dim)}
}

func (e *Embedding) Forward(ids []int) *autograd.Var { return autograd.Embedding(e.W, ids) }

func (e *Embedding) Parameters() []*autograd.Var { return []*autograd.Var{e.W} }

// Activation is an element-wise non-linearity such as autograd.Tanh or autograd.GELU.
type Activation func(*autograd.Var) *autograd.Var

// MLP is a stack of Linear layers with an activation between them (not after the last).
type MLP struct {
	Layers []*Linear
	Act    Activation
}

// NewMLP builds layers sizes[0]→sizes[1]→…→sizes[n-1]. gain is passed to hidden
// layers (√2 for ReLU-family, 5/3 for tanh, 1 otherwise); the output layer uses gain 1.
func NewMLP(rng *rand.Rand, sizes []int, act Activation, gain float64) *MLP {
	m := &MLP{Act: act}
	for i := 0; i+1 < len(sizes); i++ {
		g := gain
		if i+2 == len(sizes) {
			g = 1
		}
		m.Layers = append(m.Layers, NewLinear(rng, sizes[i], sizes[i+1], true, g))
	}
	return m
}

func (m *MLP) Forward(x *autograd.Var) *autograd.Var {
	for i, l := range m.Layers {
		x = l.Forward(x)
		if i+1 < len(m.Layers) {
			x = m.Act(x)
		}
	}
	return x
}

func (m *MLP) Parameters() []*autograd.Var {
	var ps []*autograd.Var
	for _, l := range m.Layers {
		ps = append(ps, l.Parameters()...)
	}
	return ps
}
