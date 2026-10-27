package diffusion

import (
	"fmt"
	"math"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/nn"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// NoClass is the label meaning "no condition": the model sees the extra "null" class
// slot. Training with some labels replaced by NoClass teaches one network both
// p(x | c) and p(x) — what classifier-free guidance needs.
const NoClass = -1

// TimeEmbedding maps a time τ ∈ [0, 1] to dim features (dim even), the transformer
// positional encoding applied to s = 1000·τ (as in DDPM, where t runs over 1..1000):
//
//	ω_i = 10000^(−i/(dim/2)),  emb = [sin(s·ω_0), …, sin(s·ω_{h−1}), cos(s·ω_0), …, cos(s·ω_{h−1})].
//
// High frequencies tell apart neighbouring steps, low ones give a smooth coarse "clock".
func TimeEmbedding(tau float64, dim int) []float32 {
	if dim%2 != 0 {
		panic(fmt.Sprintf("diffusion: TimeEmbedding dim %d must be even", dim))
	}
	h := dim / 2
	out := make([]float32, dim)
	s := 1000 * tau
	for i := range h {
		w := math.Exp(-math.Log(10000) * float64(i) / float64(h))
		out[i] = float32(math.Sin(s * w))
		out[h+i] = float32(math.Cos(s * w))
	}
	return out
}

// Denoiser is an MLP f(x_t, τ, c) → R². For DDPM it predicts the noise ε, for flow
// matching the velocity v; the network is the same, only the training target differs.
//
//	input row = [x, y | TimeEmbedding(τ) | one-hot(c) over NumClasses+1 slots]  →  MLP(SiLU)  →  (2)
//
// The time embedding and one-hot do not depend on parameters, so they are built as
// plain tensors and the whole trainable part is a single nn.MLP.
type Denoiser struct {
	Net        *nn.MLP
	TimeDim    int
	NumClasses int // 0 means an unconditional model (no class input)
}

// NewDenoiser builds an MLP with `layers` hidden layers of width hidden.
func NewDenoiser(rng *rand.Rand, hidden, layers, timeDim, numClasses int) *Denoiser {
	in := 2 + timeDim
	if numClasses > 0 {
		in += numClasses + 1
	}
	sizes := []int{in}
	for range layers {
		sizes = append(sizes, hidden)
	}
	sizes = append(sizes, 2)
	return &Denoiser{Net: nn.NewMLP(rng, sizes, autograd.SiLU, math.Sqrt2), TimeDim: timeDim, NumClasses: numClasses}
}

// Parameters implements nn.Module.
func (d *Denoiser) Parameters() []*autograd.Var { return d.Net.Parameters() }

// inputDim is the width of an input row.
func (d *Denoiser) inputDim() int {
	n := 2 + d.TimeDim
	if d.NumClasses > 0 {
		n += d.NumClasses + 1
	}
	return n
}

// Forward runs the network on a batch: x is B×2, tau has B times in [0, 1], labels is
// nil (unconditional) or B class ids, NoClass for the null condition. Returns (B, 2).
func (d *Denoiser) Forward(x []float32, tau []float64, labels []int) *autograd.Var {
	b := len(tau)
	if len(x) != 2*b {
		panic(fmt.Sprintf("diffusion: Forward with %d coords for %d times", len(x), b))
	}
	w := d.inputDim()
	in := make([]float32, b*w)
	for i := range b {
		row := in[i*w : (i+1)*w]
		row[0], row[1] = x[2*i], x[2*i+1]
		copy(row[2:], TimeEmbedding(tau[i], d.TimeDim))
		if d.NumClasses > 0 {
			c := d.NumClasses // null slot
			if labels != nil && labels[i] != NoClass {
				c = labels[i]
			}
			row[2+d.TimeDim+c] = 1
		}
	}
	return d.Net.Forward(autograd.NewVar(tensor.FromSlice(in, b, w)))
}

// predict is Forward for inference: it returns plain numbers. The class-free guidance
// combination is applied here when guidance != 1 and labels are given:
//
//	out = out_uncond + w·(out_cond − out_uncond)
//
// w = 1 is the plain conditional model, w = 0 the unconditional one, w > 1 pushes
// samples further towards the class.
func (d *Denoiser) predict(x []float32, tau []float64, labels []int, guidance float64) []float32 {
	cond := d.Forward(x, tau, labels).Data.Data()
	if labels == nil || d.NumClasses == 0 || guidance == 1 {
		return cond
	}
	null := make([]int, len(tau))
	for i := range null {
		null[i] = NoClass
	}
	unc := d.Forward(x, tau, null).Data.Data()
	out := make([]float32, len(cond))
	for i := range out {
		out[i] = unc[i] + float32(guidance)*(cond[i]-unc[i])
	}
	return out
}

// mse is mean((pred − target)²) over all elements — the diffusion/flow loss.
func mse(pred *autograd.Var, target []float32) *autograd.Var {
	t := autograd.NewVar(tensor.FromSlice(target, pred.Shape()...))
	return autograd.Mean(autograd.Pow(autograd.Sub(pred, t), 2))
}
