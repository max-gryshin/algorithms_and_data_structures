package autograd

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Every operation below follows the same pattern:
//   forward  — compute out.Data from the inputs;
//   backward — given out.Grad (dL/dout), add dL/dinput = dL/dout · dout/dinput to each input.

// ---------- element-wise helpers (fast paths over contiguous data) ----------

// zip applies f element-wise with broadcasting.
func zip(a, b *tensor.Tensor, f func(x, y float32) float32) *tensor.Tensor {
	as, bs := a.Shape(), b.Shape()
	if !slices.Equal(as, bs) && len(bs) > 0 && isSuffix(as, bs) {
		// Bias pattern: b repeats along the leading axes of a — index it modulo its size.
		ad, bd := a.Data(), b.Data()
		m := len(bd)
		out := make([]float32, len(ad))
		for i := range out {
			out[i] = f(ad[i], bd[i%m])
		}
		return tensor.FromSlice(out, as...)
	}
	if !slices.Equal(as, bs) {
		shape, err := tensor.BroadcastShapes(as, bs)
		if err != nil {
			panic(err)
		}
		// Materialize the broadcast (zeros + x) so the loop below is a flat one.
		if !slices.Equal(as, shape) {
			a = tensor.Add(tensor.New(shape...), a)
		}
		if !slices.Equal(bs, shape) {
			b = tensor.Add(tensor.New(shape...), b)
		}
		as = shape
	}
	ad, bd := a.Data(), b.Data()
	out := make([]float32, len(ad))
	for i := range out {
		out[i] = f(ad[i], bd[i])
	}
	return tensor.FromSlice(out, as...)
}

// mapT applies f to every element.
func mapT(a *tensor.Tensor, f func(x float32) float32) *tensor.Tensor {
	ad := a.Data()
	out := make([]float32, len(ad))
	for i, x := range ad {
		out[i] = f(x)
	}
	return tensor.FromSlice(out, a.Shape()...)
}

func mul(a, b *tensor.Tensor) *tensor.Tensor {
	return zip(a, b, func(x, y float32) float32 { return x * y })
}

// ---------- arithmetic ----------

// Add: ∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1. Broadcasting is undone by summation in accum.
func Add(a, b *Var) *Var {
	return newOp(zip(a.Data, b.Data, func(x, y float32) float32 { return x + y }), func(out *Var) {
		a.accum(out.Grad)
		b.accum(out.Grad)
	}, a, b)
}

// Sub: ∂(a−b)/∂a = 1, ∂(a−b)/∂b = −1.
func Sub(a, b *Var) *Var {
	return newOp(zip(a.Data, b.Data, func(x, y float32) float32 { return x - y }), func(out *Var) {
		a.accum(out.Grad)
		b.accum(out.Grad.Scale(-1))
	}, a, b)
}

// Mul (element-wise): ∂(a·b)/∂a = b, ∂(a·b)/∂b = a.
func Mul(a, b *Var) *Var {
	return newOp(mul(a.Data, b.Data), func(out *Var) {
		a.accum(mul(out.Grad, b.Data))
		b.accum(mul(out.Grad, a.Data))
	}, a, b)
}

// Div (element-wise): ∂(a/b)/∂a = 1/b, ∂(a/b)/∂b = −a/b².
func Div(a, b *Var) *Var {
	return newOp(zip(a.Data, b.Data, func(x, y float32) float32 { return x / y }), func(out *Var) {
		a.accum(zip(out.Grad, b.Data, func(g, y float32) float32 { return g / y }))
		if b.RequiresGrad {
			gb := zip(out.Grad, out.Data, func(g, q float32) float32 { return g * q })
			b.accum(zip(gb, b.Data, func(g, y float32) float32 { return -g / y }))
		}
	}, a, b)
}

// Scale multiplies by a constant: ∂(s·a)/∂a = s.
func Scale(a *Var, s float32) *Var {
	return newOp(a.Data.Scale(s), func(out *Var) {
		a.accum(out.Grad.Scale(s))
	}, a)
}

// Pow raises every element to a constant power: ∂(aᵖ)/∂a = p·aᵖ⁻¹.
func Pow(a *Var, p float32) *Var {
	pp := float64(p)
	return newOp(mapT(a.Data, func(x float32) float32 { return float32(math.Pow(float64(x), pp)) }), func(out *Var) {
		da := mapT(a.Data, func(x float32) float32 { return p * float32(math.Pow(float64(x), pp-1)) })
		a.accum(mul(out.Grad, da))
	}, a)
}

// ---------- unary non-linearities ----------

// unary builds an element-wise op from f and its derivative df(x, y) where y = f(x).
func unary(a *Var, f func(x float32) float32, df func(x, y float32) float32) *Var {
	return newOp(mapT(a.Data, f), func(out *Var) {
		d := zip(a.Data, out.Data, df)
		a.accum(mul(out.Grad, d))
	}, a)
}

// Exp: (eˣ)' = eˣ.
func Exp(a *Var) *Var {
	return unary(a, func(x float32) float32 { return float32(math.Exp(float64(x))) },
		func(_, y float32) float32 { return y })
}

// Log: (ln x)' = 1/x.
func Log(a *Var) *Var {
	return unary(a, func(x float32) float32 { return float32(math.Log(float64(x))) },
		func(x, _ float32) float32 { return 1 / x })
}

// Tanh: (tanh x)' = 1 − tanh²x.
func Tanh(a *Var) *Var {
	return unary(a, func(x float32) float32 { return float32(math.Tanh(float64(x))) },
		func(_, y float32) float32 { return 1 - y*y })
}

// Sigmoid: σ(x) = 1/(1+e⁻ˣ), σ' = σ(1−σ).
func Sigmoid(a *Var) *Var {
	return unary(a, sigmoid, func(_, y float32) float32 { return y * (1 - y) })
}

func sigmoid(x float32) float32 { return float32(1 / (1 + math.Exp(-float64(x)))) }

// ReLU: max(0, x); derivative 1 for x > 0, else 0.
func ReLU(a *Var) *Var {
	return unary(a, func(x float32) float32 { return max(0, x) },
		func(x, _ float32) float32 {
			if x > 0 {
				return 1
			}
			return 0
		})
}

// SiLU (a.k.a. Swish): x·σ(x); derivative σ(x)·(1 + x·(1 − σ(x))).
func SiLU(a *Var) *Var {
	return unary(a, func(x float32) float32 { return x * sigmoid(x) },
		func(x, _ float32) float32 {
			s := sigmoid(x)
			return s * (1 + x*(1-s))
		})
}

const sqrt2OverPi = 0.7978845608028654 // √(2/π)

// GELU uses the tanh approximation (GPT-2, BERT):
// 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
func GELU(a *Var) *Var {
	return unary(a, gelu, geluGrad)
}

func gelu(x float32) float32 {
	xf := float64(x)
	return float32(0.5 * xf * (1 + math.Tanh(sqrt2OverPi*(xf+0.044715*xf*xf*xf))))
}

func geluGrad(x, _ float32) float32 {
	xf := float64(x)
	u := sqrt2OverPi * (xf + 0.044715*xf*xf*xf)
	t := math.Tanh(u)
	du := sqrt2OverPi * (1 + 3*0.044715*xf*xf)
	return float32(0.5*(1+t) + 0.5*xf*(1-t*t)*du)
}

// ---------- reductions ----------

// Sum adds all elements into a scalar (shape []). The gradient of a sum is 1 for every input.
func Sum(a *Var) *Var {
	return newOp(tensor.FromSlice([]float32{float32(a.Data.Sum())}), func(out *Var) {
		a.accum(tensor.Add(tensor.New(a.Data.Shape()...), out.Grad)) // broadcast the scalar back
	}, a)
}

// Mean averages all elements into a scalar.
func Mean(a *Var) *Var { return Scale(Sum(a), 1/float32(a.Data.Size())) }

// SumLast sums over the last axis keeping it as size 1: (..., n) -> (..., 1).
func SumLast(a *Var) *Var {
	shape := a.Data.Shape()
	n := shape[len(shape)-1]
	ad := a.Data.Data()
	out := make([]float32, len(ad)/n)
	for r := range out {
		var s float64
		for _, x := range ad[r*n : (r+1)*n] {
			s += float64(x)
		}
		out[r] = float32(s)
	}
	oshape := slices.Clone(shape)
	oshape[len(oshape)-1] = 1
	return newOp(tensor.FromSlice(out, oshape...), func(o *Var) {
		a.accum(tensor.Add(tensor.New(shape...), o.Grad))
	}, a)
}

// MeanLast averages over the last axis keeping it: (..., n) -> (..., 1). Used by norms.
func MeanLast(a *Var) *Var {
	shape := a.Data.Shape()
	return Scale(SumLast(a), 1/float32(shape[len(shape)-1]))
}

// ---------- shape ----------

// MatMul is the batched, broadcasting matrix product (see tensor.MatMul).
// For C = A·B:  dA = dC·Bᵀ,  dB = Aᵀ·dC  (then summed over broadcast batch dims).
func MatMul(a, b *Var) *Var {
	if as, bs := a.Data.Shape(), b.Data.Shape(); len(as) >= 2 && len(bs) == 2 {
		return matMulShared(a, b, as, bs)
	}
	return newOp(tensor.MatMul(a.Data, b.Data), func(out *Var) {
		if a.RequiresGrad {
			a.accum(tensor.MatMul(out.Grad, b.Data.T()))
		}
		if b.RequiresGrad {
			b.accum(tensor.MatMul(a.Data.T(), out.Grad))
		}
	}, a, b)
}

// matMulShared is MatMul for (..., n, k)·(k, p): one weight matrix shared by every
// row (a Linear layer). All leading axes are flattened into one (N, k)·(k, p) product:
//
//	forward  C = A·B,   backward  dA = dC·Bᵀ (N×p · p×k),   dB = Aᵀ·dC (k×N · N×p)
//
// dB comes out already summed over the batch — no broadcast reduction is needed.
func matMulShared(a, b *Var, as, bs []int) *Var {
	k, p := bs[0], bs[1]
	if as[len(as)-1] != k {
		panic(fmt.Sprintf("autograd: MatMul inner dims differ: %v x %v", as, bs))
	}
	n := a.Data.Size() / k
	oshape := append(slices.Clone(as[:len(as)-1]), p)
	out := mm(a.Data.Data(), b.Data.Data(), n, k, p)
	return newOp(tensor.FromSlice(out, oshape...), func(o *Var) {
		g := o.Grad.Data()
		if a.RequiresGrad {
			a.accum(tensor.FromSlice(mmBT(g, b.Data.Data(), n, p, k), as...))
		}
		if b.RequiresGrad {
			b.accum(tensor.FromSlice(mmAT(a.Data.Data(), g, n, k, p), k, p))
		}
	}, a, b)
}

// MatMulBT computes a·bᵀ for a (..., n, k) and a 2-D b (p, k) without materializing bᵀ.
// It is the tied LM head of GPT-2: logits = x·Eᵀ with the embedding table E (V, d).
//
//	forward  C = A·Bᵀ,   backward  dA = dC·B (N×p · p×k),   dB = dCᵀ·A (p×N · N×k)
func MatMulBT(a, b *Var) *Var {
	as, bs := a.Data.Shape(), b.Data.Shape()
	if len(bs) != 2 || as[len(as)-1] != bs[1] {
		panic(fmt.Sprintf("autograd: MatMulBT %v x %vᵀ", as, bs))
	}
	p, k := bs[0], bs[1]
	n := a.Data.Size() / k
	oshape := append(slices.Clone(as[:len(as)-1]), p)
	out := mmBT(a.Data.Data(), b.Data.Data(), n, k, p)
	return newOp(tensor.FromSlice(out, oshape...), func(o *Var) {
		g := o.Grad.Data()
		if a.RequiresGrad {
			a.accum(tensor.FromSlice(mm(g, b.Data.Data(), n, p, k), as...))
		}
		if b.RequiresGrad {
			b.accum(tensor.FromSlice(mmAT(g, a.Data.Data(), n, p, k), p, k))
		}
	}, a, b)
}

// Reshape changes the shape; the gradient is reshaped back.
func Reshape(a *Var, shape ...int) *Var {
	orig := a.Data.Shape()
	return newOp(a.Data.Reshape(shape...), func(out *Var) {
		a.accum(out.Grad.Reshape(orig...))
	}, a)
}

// Transpose swaps two axes; the gradient is transposed back.
func Transpose(a *Var, d1, d2 int) *Var {
	return newOp(transposeData(a.Data, d1, d2), func(out *Var) {
		a.accum(transposeData(out.Grad, d1, d2))
	}, a)
}

// Embedding gathers rows of weight (V, C) by ids and returns (len(ids), C).
// It equals onehot(ids)·W, so the gradient of row i is the sum of the output
// gradients of all positions where id == i (scatter-add).
func Embedding(weight *Var, ids []int) *Var {
	ws := weight.Data.Shape()
	v, c := ws[0], ws[1]
	wd := weight.Data.Data()
	out := make([]float32, len(ids)*c)
	for r, id := range ids {
		if id < 0 || id >= v {
			panic(fmt.Sprintf("autograd: Embedding id %d out of range [0,%d)", id, v))
		}
		copy(out[r*c:(r+1)*c], wd[id*c:(id+1)*c])
	}
	return newOp(tensor.FromSlice(out, len(ids), c), func(o *Var) {
		g := tensor.New(v, c)
		gd, od := g.Data(), o.Grad.Data()
		for r, id := range ids {
			row := gd[id*c : (id+1)*c]
			for j, x := range od[r*c : (r+1)*c] {
				row[j] += x
			}
		}
		weight.accum(g)
	}, weight)
}

// ---------- softmax and loss ----------

// Softmax over the last axis (numerically stable: the row max is subtracted first).
// Backward: dx = y ⊙ (dy − Σ(dy ⊙ y)), the sum taken over the same row.
func Softmax(a *Var) *Var {
	shape := a.Data.Shape()
	n := shape[len(shape)-1]
	ad := a.Data.Data()
	out := make([]float32, len(ad))
	for r := 0; r < len(ad); r += n {
		softmaxRow(ad[r:r+n], out[r:r+n])
	}
	return newOp(tensor.FromSlice(out, shape...), func(o *Var) {
		yd, gd := o.Data.Data(), o.Grad.Data()
		dx := make([]float32, len(yd))
		for r := 0; r < len(yd); r += n {
			var dot float64
			for j := r; j < r+n; j++ {
				dot += float64(gd[j]) * float64(yd[j])
			}
			for j := r; j < r+n; j++ {
				dx[j] = yd[j] * (gd[j] - float32(dot))
			}
		}
		a.accum(tensor.FromSlice(dx, shape...))
	}, a)
}

func softmaxRow(z, out []float32) {
	m := float32(math.Inf(-1))
	for _, x := range z {
		m = max(m, x)
	}
	var s float64
	for j, x := range z {
		e := math.Exp(float64(x - m))
		out[j] = float32(e)
		s += e
	}
	for j := range out {
		out[j] = float32(float64(out[j]) / s)
	}
}

// CrossEntropy is the mean negative log-likelihood of targets under softmax(logits).
// logits: (..., V), flattened to N rows; targets: N class ids. A target < 0 is ignored
// (used later to mask prompt tokens in SFT). Fused for stability and speed:
//
//	loss_i = logsumexp(z_i) − z_i[t_i],   d loss_i / d z_i = softmax(z_i) − onehot(t_i).
func CrossEntropy(logits *Var, targets []int) *Var {
	shape := logits.Data.Shape()
	v := shape[len(shape)-1]
	zd := logits.Data.Data()
	if len(zd)/v != len(targets) {
		panic(fmt.Sprintf("autograd: CrossEntropy logits %v vs %d targets", shape, len(targets)))
	}
	probs := make([]float32, len(zd))
	var total float64
	count := 0
	for r, t := range targets {
		row := zd[r*v : (r+1)*v]
		softmaxRow(row, probs[r*v:(r+1)*v])
		if t < 0 {
			continue
		}
		count++
		total -= math.Log(math.Max(float64(probs[r*v+t]), 1e-30))
	}
	if count == 0 {
		count = 1
	}
	loss := tensor.FromSlice([]float32{float32(total / float64(count))})
	return newOp(loss, func(o *Var) {
		scale := o.Grad.Data()[0] / float32(count)
		g := make([]float32, len(zd))
		for r, t := range targets {
			if t < 0 {
				continue
			}
			for j := r * v; j < (r+1)*v; j++ {
				g[j] = probs[j] * scale
			}
			g[r*v+t] -= scale
		}
		logits.accum(tensor.FromSlice(g, shape...))
	}, logits)
}

// Dropout zeroes each element with probability p and scales the rest by 1/(1−p),
// so the expected value is unchanged. Call it only in training mode.
func Dropout(a *Var, p float32, rng *rand.Rand) *Var {
	if p <= 0 {
		return a
	}
	mask := mapT(a.Data, func(float32) float32 {
		if rng.Float32() < p {
			return 0
		}
		return 1 / (1 - p)
	})
	return newOp(mul(a.Data, mask), func(out *Var) {
		a.accum(mul(out.Grad, mask))
	}, a)
}
