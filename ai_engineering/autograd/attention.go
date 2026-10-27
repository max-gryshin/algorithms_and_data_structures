package autograd

import (
	"fmt"
	"slices"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// CausalAttention is fused scaled dot-product attention with a causal mask and
// grouped K/V heads (lectures L7, L9, L10):
//
//	S = Q·Kᵀ·scale,   P = softmax(S + mask),   O = P·V
//
// q: (B, h, Tq, d); k, v: (B, h_kv, Tk, d) with h_kv | h — query head i reads K/V head
// i / (h/h_kv). Queries are the last Tq of the Tk positions (bottom-right alignment,
// like nn.CausalMask): query r sees keys 0 … Tk−Tq+r. Tq < Tk happens with a KV-cache.
//
// It returns O (B, h, Tq, d) and the probabilities P (B, h, Tq, Tk) for inspection.
// The composite version (MatMul, Scale, Add mask, Softmax, MatMul) builds five
// (B, h, T, T) tensors and a broadcast K/V copy; the fused one makes one pass per
// query row and splits (batch, K/V head) pairs between cores. Backward (L7):
//
//	dP = dO·Vᵀ,   dS = P ⊙ (dP − rowsum(dP ⊙ P)),   dQ = dS·K·scale,
//	dK = dSᵀ·Q·scale,   dV = Pᵀ·dO.
func CausalAttention(q, k, v *Var, scale float32) (*Var, *tensor.Tensor) {
	qs, ks := q.Data.Shape(), k.Data.Shape()
	if len(qs) != 4 || len(ks) != 4 || qs[0] != ks[0] || qs[3] != ks[3] || qs[1]%ks[1] != 0 ||
		qs[2] > ks[2] || !slices.Equal(ks, v.Data.Shape()) {
		panic(fmt.Sprintf("autograd: CausalAttention q %v k %v v %v", qs, ks, v.Data.Shape()))
	}
	b, h, tq, d := qs[0], qs[1], qs[2], qs[3]
	kvh, tk := ks[1], ks[2]
	g := h / kvh
	qd, kd, vd := q.Data.Data(), k.Data.Data(), v.Data.Data()
	out := make([]float32, b*h*tq*d)
	probs := make([]float32, b*h*tq*tk)

	units := b * kvh // one unit = one K/V head of one sequence and its g query heads
	parallelFor(units, b*h*tq*tk*d*2, func(lo, hi int) {
		for u := lo; u < hi; u++ {
			kv := u * tk * d // offset of this K/V head
			for gi := range g {
				head := u*g + gi // index into (B·h): u = bi·kvh + j → head = bi·h + j·g + gi
				for r := range tq {
					row := head*tq + r
					AttendRow(qd[row*d:(row+1)*d], kd[kv:kv+tk*d], vd[kv:kv+tk*d], tk-tq+r+1, d, scale,
						probs[row*tk:(row+1)*tk], out[row*d:(row+1)*d])
				}
			}
		}
	})
	pt := tensor.FromSlice(probs, b, h, tq, tk)
	return newOp(tensor.FromSlice(out, b, h, tq, d), func(o *Var) {
		do := o.Grad.Data()
		dq := make([]float32, len(qd))
		dk := make([]float32, len(kd))
		dv := make([]float32, len(vd))
		parallelFor(units, b*h*tq*tk*d*4, func(lo, hi int) {
			dp := make([]float32, tk)
			for u := lo; u < hi; u++ {
				kv := u * tk * d
				for gi := range g {
					head := u*g + gi
					for r := range tq {
						row := head*tq + r
						n := tk - tq + r + 1
						p := probs[row*tk : row*tk+n]
						dor, qr := do[row*d:(row+1)*d], qd[row*d:(row+1)*d]
						var s float64 // rowsum(dP ⊙ P)
						for j := range n {
							dp[j] = dot(dor, vd[kv+j*d:kv+(j+1)*d])
							s += float64(dp[j]) * float64(p[j])
						}
						dqr := dq[row*d : (row+1)*d]
						for j := range n {
							ds := p[j] * (dp[j] - float32(s)) * scale
							kj, dkj, dvj := kd[kv+j*d:kv+(j+1)*d], dk[kv+j*d:kv+(j+1)*d], dv[kv+j*d:kv+(j+1)*d]
							for c := range d {
								dqr[c] += ds * kj[c]
								dkj[c] += ds * qr[c]
								dvj[c] += p[j] * dor[c]
							}
						}
					}
				}
			}
		})
		q.accum(tensor.FromSlice(dq, qs...))
		k.accum(tensor.FromSlice(dk, ks...))
		v.accum(tensor.FromSlice(dv, ks...))
	}, q, k, v), pt
}

// AttendRow computes one query row of attention over the first n keys:
//
//	s_j = (q·k_j)·scale,  p = softmax(s_0 … s_{n−1}),  out = Σ_j p_j·v_j
//
// keys and values are row-major (≥ n rows of d); probs receives p (entries ≥ n are
// zeroed) and out the result. Both the training forward and the KV-cache decode
// (L10) call this one function, so cached and uncached logits agree bit for bit.
func AttendRow(q, keys, values []float32, n, d int, scale float32, probs, out []float32) {
	for j := range n {
		probs[j] = dot(q, keys[j*d:(j+1)*d]) * scale
	}
	softmaxRow(probs[:n], probs[:n])
	clear(probs[n:])
	clear(out)
	for j := range n {
		pj := probs[j]
		vj := values[j*d : (j+1)*d]
		for c := range d {
			out[c] += pj * vj[c]
		}
	}
}
