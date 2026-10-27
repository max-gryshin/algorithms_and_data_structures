// Package serve is the inference server of lecture L14: tiled attention with online
// softmax (FlashAttention), a paged KV-cache block allocator, a radix-tree prefix cache,
// a continuous-batching scheduler and an OpenAI-compatible HTTP API with SSE streaming.
package serve

import (
	"math"
)

// NaiveAttention computes softmax(q·kᵀ/√d + mask)·v for one head and materializes the
// whole (tq, tk) score matrix. q: (tq, d), k, v: (tk, d), row-major. Causal masking is
// bottom-right aligned (like nn.CausalMask): query i sits at position tk−tq+i.
// scratch is the number of floats of temporary memory: tq·tk.
func NaiveAttention(q, k, v []float32, tq, tk, d int, causal bool) (out []float32, scratch int) {
	scale := 1 / math.Sqrt(float64(d))
	s := make([]float64, tq*tk)
	out = make([]float32, tq*d)
	for i := range tq {
		row := s[i*tk : (i+1)*tk]
		m := math.Inf(-1)
		for j := range tk {
			if causal && j > tk-tq+i {
				row[j] = math.Inf(-1)
				continue
			}
			row[j] = dot(q[i*d:(i+1)*d], k[j*d:(j+1)*d]) * scale
			m = max(m, row[j])
		}
		var l float64
		for j := range row {
			row[j] = math.Exp(row[j] - m)
			l += row[j]
		}
		for j := range tk {
			p := row[j] / l
			for c := range d {
				out[i*d+c] += float32(p * float64(v[j*d+c]))
			}
		}
	}
	return out, tq * tk
}

// FlashAttention is NaiveAttention computed tile by tile (Dao et al., 2022): bq queries
// against bk keys at a time with an online softmax — a running maximum m and sum l per
// query; when a new tile raises the maximum, the old sum and accumulator are rescaled by
// e^(m_old − m_new). The (tq, tk) matrix never exists: scratch is bq·bk + (d+2)·bq floats,
// independent of the sequence length.
func FlashAttention(q, k, v []float32, tq, tk, d, bq, bk int, causal bool) (out []float32, scratch int) {
	scale := 1 / math.Sqrt(float64(d))
	out = make([]float32, tq*d)
	s := make([]float64, bq*bk)
	m := make([]float64, bq)
	l := make([]float64, bq)
	acc := make([]float64, bq*d)
	for i0 := 0; i0 < tq; i0 += bq {
		i1 := min(i0+bq, tq)
		for i := range bq {
			m[i], l[i] = math.Inf(-1), 0
		}
		clear(acc)
		for j0 := 0; j0 < tk; j0 += bk {
			j1 := min(j0+bk, tk)
			if causal && j0 > tk-tq+i1-1 {
				break // the whole tile lies above the diagonal
			}
			for i := i0; i < i1; i++ {
				ii := i - i0
				hi := j1
				if causal {
					hi = min(j1, tk-tq+i+1)
				}
				if hi <= j0 {
					continue
				}
				st := s[ii*bk : ii*bk+hi-j0]
				mt := math.Inf(-1)
				for j := j0; j < hi; j++ {
					st[j-j0] = dot(q[i*d:(i+1)*d], k[j*d:(j+1)*d]) * scale
					mt = max(mt, st[j-j0])
				}
				mNew := max(m[ii], mt)
				alpha := math.Exp(m[ii] - mNew) // rescales everything summed so far
				a := acc[ii*d : (ii+1)*d]
				for c := range a {
					a[c] *= alpha
				}
				l[ii] *= alpha
				for j := j0; j < hi; j++ {
					p := math.Exp(st[j-j0] - mNew)
					l[ii] += p
					for c := range d {
						a[c] += p * float64(v[j*d+c])
					}
				}
				m[ii] = mNew
			}
		}
		for i := i0; i < i1; i++ {
			ii := i - i0
			for c := range d {
				out[i*d+c] = float32(acc[ii*d+c] / l[ii])
			}
		}
	}
	return out, bq*bk + (d+2)*bq
}

// OnlineSoftmax is the running state (max M, normalizer L) of a softmax over a stream of
// logits that arrives in blocks.
type OnlineSoftmax struct{ M, L float64 }

// NewOnlineSoftmax starts with M = −∞, L = 0.
func NewOnlineSoftmax() OnlineSoftmax { return OnlineSoftmax{M: math.Inf(-1)} }

// Push adds a block of logits and returns α = e^(M_old − M_new), the factor by which
// everything accumulated before had to be rescaled.
func (s *OnlineSoftmax) Push(block []float64) (alpha float64) {
	mt := math.Inf(-1)
	for _, x := range block {
		mt = max(mt, x)
	}
	mNew := max(s.M, mt)
	alpha = math.Exp(s.M - mNew)
	s.L *= alpha
	for _, x := range block {
		s.L += math.Exp(x - mNew)
	}
	s.M = mNew
	return alpha
}

// Merge combines two partial states (split-K / Flash-Decoding).
func (s OnlineSoftmax) Merge(o OnlineSoftmax) OnlineSoftmax {
	m := max(s.M, o.M)
	return OnlineSoftmax{M: m, L: s.L*math.Exp(s.M-m) + o.L*math.Exp(o.M-m)}
}

func dot(a, b []float32) float64 {
	var s float64
	for i := range a {
		s += float64(a[i]) * float64(b[i])
	}
	return s
}
