package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Token and position embeddings (lecture L6).

// NewLinearStd is NewLinear with an explicit weight std: W ~ N(0, std²), b = 0.
// GPT-2 initializes every matrix with std 0.02 instead of 1/√in.
func NewLinearStd(rng *rand.Rand, in, out int, bias bool, std float32) *Linear {
	l := &Linear{W: Normal(rng, "linear.W", std, in, out)}
	if bias {
		l.B = Zeros("linear.b", out)
	}
	return l
}

// ForwardBatch looks up a batch of sequences: ids (B, T) → (B, T, Dim).
// All rows must have the same length.
func (e *Embedding) ForwardBatch(ids [][]int) *autograd.Var {
	b, t := len(ids), len(ids[0])
	flat := make([]int, 0, b*t)
	for _, row := range ids {
		if len(row) != t {
			panic(fmt.Sprintf("nn: ForwardBatch ragged batch: rows of %d and %d tokens", t, len(row)))
		}
		flat = append(flat, row...)
	}
	x := e.Forward(flat)                           // (B·T, Dim)
	return autograd.Reshape(x, b, t, x.Shape()[1]) // (B, T, Dim)
}

// Positions returns ids pos, pos+1, …, pos+t−1 — the absolute positions of t tokens
// starting at offset pos (pos > 0 when earlier tokens are already in a KV-cache, L10).
func Positions(pos, t int) []int {
	p := make([]int, t)
	for i := range p {
		p[i] = pos + i
	}
	return p
}

// SinusoidalPE returns the fixed table of Vaswani et al. (2017), shape (maxT, dim):
//
//	PE[p, 2i]   = sin(p · ω_i),   PE[p, 2i+1] = cos(p · ω_i),   ω_i = 10000^(−2i/dim)
//
// dim must be even. Each pair (sin, cos) is a point on a circle rotating with
// angular speed ω_i, so PE[p+k] is a fixed rotation of PE[p] (proof in L6).
func SinusoidalPE(maxT, dim int) *tensor.Tensor {
	if dim%2 != 0 {
		panic("nn: SinusoidalPE needs an even dim")
	}
	pe := tensor.New(maxT, dim)
	d := pe.Data()
	for p := range maxT {
		for i := range dim / 2 {
			w := math.Pow(10000, -float64(2*i)/float64(dim))
			d[p*dim+2*i] = float32(math.Sin(float64(p) * w))
			d[p*dim+2*i+1] = float32(math.Cos(float64(p) * w))
		}
	}
	return pe
}

// Rows returns rows [from, to) of a 2-D tensor as a new contiguous tensor.
func Rows(t *tensor.Tensor, from, to int) *tensor.Tensor {
	s := t.Shape()
	if len(s) != 2 || from < 0 || to > s[0] || from > to {
		panic(fmt.Sprintf("nn: Rows [%d, %d) of %v", from, to, s))
	}
	d := t.Data()
	out := make([]float32, (to-from)*s[1])
	copy(out, d[from*s[1]:to*s[1]])
	return tensor.FromSlice(out, to-from, s[1])
}

// RoPE holds precomputed cos/sin tables for rotary position embeddings:
// position p, frequency i → angle θ = (p / Scale) · Theta^(−2i/HeadDim).
type RoPE struct {
	HeadDim int
	MaxT    int
	Theta   float64 // base: 10 000 (RoFormer, Llama 1/2), 500 000 (Llama 3)
	Scale   float64 // position interpolation factor (1 = none; 4 squeezes 4× longer contexts into the trained range)
	Cos     *tensor.Tensor
	Sin     *tensor.Tensor // both (MaxT, HeadDim/2)
}

// NewRoPE precomputes tables for positions 0..maxT−1.
func NewRoPE(headDim, maxT int, theta float64) *RoPE {
	return NewRoPEInterpolated(headDim, maxT, theta, 1)
}

// NewRoPEInterpolated is RoPE with position interpolation (Chen et al., 2023):
// positions are divided by scale, so a model trained on N tokens sees angles
// from its trained range for sequences of up to scale·N tokens.
func NewRoPEInterpolated(headDim, maxT int, theta, scale float64) *RoPE {
	if headDim%2 != 0 {
		panic("nn: RoPE needs an even head dim")
	}
	r := &RoPE{HeadDim: headDim, MaxT: maxT, Theta: theta, Scale: scale,
		Cos: tensor.New(maxT, headDim/2), Sin: tensor.New(maxT, headDim/2)}
	cd, sd := r.Cos.Data(), r.Sin.Data()
	for p := range maxT {
		for i := range headDim / 2 {
			w := math.Pow(theta, -float64(2*i)/float64(headDim))
			a := float64(p) / scale * w
			cd[p*headDim/2+i] = float32(math.Cos(a))
			sd[p*headDim/2+i] = float32(math.Sin(a))
		}
	}
	return r
}

// Apply rotates x: (..., T, HeadDim) whose rows sit at absolute positions pos..pos+T−1.
func (r *RoPE) Apply(x *autograd.Var, pos int) *autograd.Var {
	s := x.Shape()
	t := s[len(s)-2]
	if pos+t > r.MaxT {
		panic(fmt.Sprintf("nn: RoPE positions up to %d exceed table of %d", pos+t, r.MaxT))
	}
	return autograd.Rotary(x, Rows(r.Cos, pos, pos+t), Rows(r.Sin, pos, pos+t))
}
