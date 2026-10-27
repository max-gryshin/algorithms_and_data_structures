package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// LoRA is a low-rank adapter of a Linear layer (Hu et al., 2021; lecture L13):
//
//	y = x·W + b + (α/r)·(x·A)·B,   A: (In, r) ~ N(0, 1/In),  B: (r, Out) = 0.
//
// In the paper's column layout (h = W₀x) the update is ΔW = B·A with B (d×r), A (r×k);
// our Linear stores W as (In, Out) and multiplies rows, so ΔW = A·B (In×r · r×Out) — the
// same thing transposed: A still "reads" the input down to r numbers, B "writes" them
// up to the output. B starts at zero, so at step 0 the adapted model is exactly the
// base model and the first gradient of B is non-zero (it is (x·A)ᵀ·dy, and x·A ≠ 0).
//
// Trainable parameters: r·(In + Out) instead of In·Out.
type LoRA struct {
	A, B   *autograd.Var
	Rank   int
	Alpha  float32
	Merged bool // ΔW has been added into the base W; Forward skips the adapter
}

// Scale is α/r: the multiplier of the low-rank branch. Keeping α fixed while changing r
// keeps the size of the update roughly constant, so the learning rate need not be retuned.
func (a *LoRA) Scale() float32 { return a.Alpha / float32(a.Rank) }

// Forward is the adapter branch alone: (α/r)·(x·A)·B, x: (..., In) → (..., Out).
func (a *LoRA) Forward(x *autograd.Var) *autograd.Var {
	h := autograd.MatMul(x, a.A)                              // (..., r): the "bottleneck"
	return autograd.Scale(autograd.MatMul(h, a.B), a.Scale()) // (..., Out)
}

func (a *LoRA) Parameters() []*autograd.Var { return []*autograd.Var{a.A, a.B} }

// Delta is the dense update ΔW = (α/r)·A·B, shape (In, Out).
func (a *LoRA) Delta() *tensor.Tensor { return tensor.MatMul(a.A.Data, a.B.Data).Scale(a.Scale()) }

// AddLoRA attaches a fresh adapter of rank r to l and returns it.
func (l *Linear) AddLoRA(rng *rand.Rand, rank int, alpha float32) *LoRA {
	in, out := l.W.Data.Shape()[0], l.W.Data.Shape()[1]
	if rank <= 0 || rank > min(in, out) {
		panic(fmt.Sprintf("nn: LoRA rank %d for a %d×%d matrix", rank, in, out))
	}
	l.LoRA = &LoRA{
		A:     Normal(rng, "lora.A", float32(1/math.Sqrt(float64(in))), in, rank),
		B:     Zeros("lora.B", rank, out),
		Rank:  rank,
		Alpha: alpha,
	}
	return l.LoRA
}

// Merge adds ΔW into W in place: after it the layer costs exactly as much as without
// an adapter (no extra latency at inference) and Forward gives the same result.
func (l *Linear) Merge() {
	if l.LoRA == nil || l.LoRA.Merged {
		return
	}
	addInto(l.W.Data, l.LoRA.Delta(), 1)
	l.LoRA.Merged = true
}

// Unmerge subtracts ΔW again (to keep training the adapter or swap it for another).
// In float32 W + ΔW − ΔW may differ from W in the last bit.
func (l *Linear) Unmerge() {
	if l.LoRA == nil || !l.LoRA.Merged {
		return
	}
	addInto(l.W.Data, l.LoRA.Delta(), -1)
	l.LoRA.Merged = false
}

func addInto(dst, src *tensor.Tensor, sign float32) {
	d, s := dst.Data(), src.Data()
	for i := range d {
		d[i] += sign * s[i]
	}
}
