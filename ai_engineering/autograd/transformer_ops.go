package autograd

import (
	"fmt"
	"math"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Fused operations for the transformer (lectures L6–L8). Each of them could be
// composed from the primitive ops above, but a fused forward/backward is both
// faster (one pass over memory instead of five) and a good exercise: its
// gradient is derived by hand in the lectures and verified with GradCheck.

// LayerNorm normalizes every row of the last axis to zero mean and unit variance,
// then applies a per-feature gain and (optional) bias:
//
//	μ = mean(x),  σ² = mean((x − μ)²),  x̂ = (x − μ)/√(σ² + eps),  y = g ⊙ x̂ + b
//
// x: (..., n); g, b: (n). b may be nil (LayerNorm without bias).
// Backward (L8):  dx̂ = dy ⊙ g,  dx = (dx̂ − mean(dx̂) − x̂ ⊙ mean(dx̂ ⊙ x̂)) / √(σ² + eps),
// dg = Σ_rows dy ⊙ x̂,  db = Σ_rows dy.
func LayerNorm(x, g, b *Var, eps float32) *Var {
	return norm(x, g, b, eps, true)
}

// RMSNorm divides every row of the last axis by its root mean square, then applies a gain
// (Zhang & Sennrich, 2019; used by Llama):
//
//	r = √(mean(x²) + eps),  x̂ = x / r,  y = g ⊙ x̂
//
// x: (..., n); g: (n). Backward (L8):  dx = (dx̂ − x̂ ⊙ mean(dx̂ ⊙ x̂)) / r,  dx̂ = dy ⊙ g.
func RMSNorm(x, g *Var, eps float32) *Var {
	return norm(x, g, nil, eps, false)
}

// norm implements both LayerNorm (center = true) and RMSNorm (center = false).
func norm(x, g, b *Var, eps float32, center bool) *Var {
	shape := x.Data.Shape()
	n := shape[len(shape)-1]
	if gs := g.Data.Shape(); len(gs) != 1 || gs[0] != n {
		panic(fmt.Sprintf("autograd: norm gain %v for input %v", gs, shape))
	}
	xd, gd := x.Data.Data(), g.Data.Data()
	var bd []float32
	if b != nil {
		bd = b.Data.Data()
	}
	rows := len(xd) / n
	xhat := make([]float32, len(xd)) // saved for backward
	inv := make([]float64, rows)     // 1/σ (LayerNorm) or 1/r (RMSNorm) per row
	out := make([]float32, len(xd))
	for r := range rows {
		row := xd[r*n : (r+1)*n]
		var mean float64
		if center {
			for _, v := range row {
				mean += float64(v)
			}
			mean /= float64(n)
		}
		var ss float64
		for _, v := range row {
			c := float64(v) - mean
			ss += c * c
		}
		inv[r] = 1 / math.Sqrt(ss/float64(n)+float64(eps))
		for j, v := range row {
			h := float32((float64(v) - mean) * inv[r])
			xhat[r*n+j] = h
			y := h * gd[j]
			if bd != nil {
				y += bd[j]
			}
			out[r*n+j] = y
		}
	}
	inputs := []*Var{x, g}
	if b != nil {
		inputs = append(inputs, b)
	}
	return newOp(tensor.FromSlice(out, shape...), func(o *Var) {
		dy := o.Grad.Data()
		if g.RequiresGrad {
			dg := make([]float32, n)
			for i, d := range dy {
				dg[i%n] += d * xhat[i]
			}
			g.accum(tensor.FromSlice(dg, n))
		}
		if b != nil && b.RequiresGrad {
			db := make([]float32, n)
			for i, d := range dy {
				db[i%n] += d
			}
			b.accum(tensor.FromSlice(db, n))
		}
		if !x.RequiresGrad {
			return
		}
		dx := make([]float32, len(dy))
		for r := range rows {
			var m1, m2 float64 // mean(dx̂), mean(dx̂ ⊙ x̂)
			for j := range n {
				dh := float64(dy[r*n+j] * gd[j])
				m1 += dh
				m2 += dh * float64(xhat[r*n+j])
			}
			m1 /= float64(n)
			m2 /= float64(n)
			if !center {
				m1 = 0
			}
			for j := range n {
				dh := float64(dy[r*n+j] * gd[j])
				dx[r*n+j] = float32((dh - m1 - float64(xhat[r*n+j])*m2) * inv[r])
			}
		}
		x.accum(tensor.FromSlice(dx, shape...))
	}, inputs...)
}

// Rotary applies the rotary position embedding (RoPE, Su et al. 2021, lecture L6).
// x: (..., T, d) with even d; cos, sin: (T, d/2) hold cos θ and sin θ for every
// position t and frequency i. Each pair (x[2i], x[2i+1]) of row t is rotated by θ[t,i]:
//
//	y[2i]   = x[2i]·cos θ − x[2i+1]·sin θ
//	y[2i+1] = x[2i]·sin θ + x[2i+1]·cos θ
//
// A rotation is orthogonal, so the backward pass is the rotation by −θ (the transpose).
// Pairs are adjacent elements, as in RoFormer and Meta's Llama code; Hugging Face
// pairs element i with i + d/2 instead (the "rotate_half" layout) — same math, permuted dims.
func Rotary(x *Var, cos, sin *tensor.Tensor) *Var {
	shape := x.Data.Shape()
	if len(shape) < 2 {
		panic(fmt.Sprintf("autograd: Rotary needs (..., T, d), got %v", shape))
	}
	t, d := shape[len(shape)-2], shape[len(shape)-1]
	if cs := cos.Shape(); d%2 != 0 || len(cs) != 2 || cs[0] != t || cs[1] != d/2 {
		panic(fmt.Sprintf("autograd: Rotary input %v vs table %v", shape, cos.Shape()))
	}
	cd, sd := cos.Data(), sin.Data()
	rotate := func(src []float32, sign float32) []float32 {
		dst := make([]float32, len(src))
		for r := 0; r < len(src)/d; r++ {
			tab := (r % t) * (d / 2) // position of row r inside its sequence
			row, out := src[r*d:(r+1)*d], dst[r*d:(r+1)*d]
			for i := range d / 2 {
				c, s := cd[tab+i], sign*sd[tab+i]
				x0, x1 := row[2*i], row[2*i+1]
				out[2*i] = x0*c - x1*s
				out[2*i+1] = x0*s + x1*c
			}
		}
		return dst
	}
	return newOp(tensor.FromSlice(rotate(x.Data.Data(), 1), shape...), func(o *Var) {
		x.accum(tensor.FromSlice(rotate(o.Grad.Data(), -1), shape...))
	}, x)
}
