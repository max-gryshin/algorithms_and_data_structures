package nn

import (
	"fmt"
	"math"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// KVCache stores the keys and values of one attention layer for B sequences
// (lecture L10). Keys are stored after RoPE, so a cached key never has to be rotated
// again. Queries are not stored: a query is used once, by its own token.
//
// Layout: (B, KVHeads, Cap, HeadDim) — the rows of one head are contiguous, so the
// decode step reads every key with a sequential scan (memory-bound, L10).
type KVCache struct {
	B, KVHeads, HeadDim, Cap int
	Len                      int       // tokens cached so far (the same for every sequence)
	K, V                     []float32 // B·KVHeads·Cap·HeadDim each
}

// NewKVCache allocates a cache for up to capacity tokens.
func NewKVCache(b, kvHeads, headDim, capacity int) *KVCache {
	n := b * kvHeads * capacity * headDim
	return &KVCache{B: b, KVHeads: kvHeads, HeadDim: headDim, Cap: capacity,
		K: make([]float32, n), V: make([]float32, n)}
}

// Bytes is the memory of the filled part in float32: 2·B·h_kv·Len·d_head·4.
func (c *KVCache) Bytes() int { return 2 * c.B * c.KVHeads * c.Len * c.HeadDim * 4 }

// Truncate forgets every token from position n on (speculative decoding rolls back
// rejected draft tokens this way; Reset is Truncate(0)).
func (c *KVCache) Truncate(n int) {
	if n < 0 || n > c.Len {
		panic(fmt.Sprintf("nn: KVCache.Truncate(%d) with %d cached", n, c.Len))
	}
	c.Len = n
}

// head returns the K and V rows of sequence bi, K/V head j.
func (c *KVCache) head(bi, j int) (k, v []float32) {
	off := (bi*c.KVHeads + j) * c.Cap * c.HeadDim
	return c.K[off : off+c.Cap*c.HeadDim], c.V[off : off+c.Cap*c.HeadDim]
}

// ForwardCached is attention for x: (B, Tn, C) — the Tn newest tokens, at positions
// c.Len … c.Len+Tn−1. Their keys and values are appended to c; every query attends to
// all cached keys plus the new ones up to itself. Inference only: the attention
// output is a constant (no gradient flows into the cache).
//
// Every query row goes through autograd.AttendRow — the same function the training
// forward uses — so the logits equal those of a full Forward bit for bit.
func (a *CausalSelfAttention) ForwardCached(x *autograd.Var, c *KVCache) *autograd.Var {
	s := x.Shape()
	b, tn := s[0], s[1]
	h, kv, d := a.Heads, a.KVHeads, a.HeadDim
	if b != c.B || kv != c.KVHeads || d != c.HeadDim {
		panic(fmt.Sprintf("nn: cache (B=%d, h_kv=%d, d=%d) for attention (B=%d, h_kv=%d, d=%d)",
			c.B, c.KVHeads, c.HeadDim, b, kv, d))
	}
	pos := c.Len
	if pos+tn > c.Cap {
		panic(fmt.Sprintf("nn: KV-cache full: %d cached + %d new > capacity %d", pos, tn, c.Cap))
	}
	split := func(y *autograd.Var, heads int) *autograd.Var {
		return autograd.Transpose(autograd.Reshape(y, b, tn, heads, d), 1, 2) // (B, heads, Tn, d)
	}
	q := split(a.Q.Forward(x), h)
	k := split(a.K.Forward(x), kv)
	v := split(a.V.Forward(x), kv)
	if a.Rope != nil {
		q = a.Rope.Apply(q, pos)
		k = a.Rope.Apply(k, pos)
	}
	kd, vd, qd := k.Data.Data(), v.Data.Data(), q.Data.Data()
	for bi := range b { // append the new keys/values: rows pos … pos+Tn−1 of every head
		for j := range kv {
			ck, cv := c.head(bi, j)
			src := (bi*kv + j) * tn * d
			copy(ck[pos*d:(pos+tn)*d], kd[src:src+tn*d])
			copy(cv[pos*d:(pos+tn)*d], vd[src:src+tn*d])
		}
	}
	c.Len += tn

	g := h / kv
	scale := float32(1 / math.Sqrt(float64(d)))
	out := make([]float32, b*h*tn*d)
	probs := make([]float32, c.Len)
	for bi := range b {
		for hi := range h {
			ck, cv := c.head(bi, hi/g)
			for r := range tn {
				row := (bi*h+hi)*tn + r
				autograd.AttendRow(qd[row*d:(row+1)*d], ck, cv, pos+r+1, d, scale, probs, out[row*d:(row+1)*d])
			}
		}
	}
	y := autograd.NewVar(tensor.FromSlice(out, b, h, tn, d))
	y = autograd.Reshape(autograd.Transpose(y, 1, 2), b, tn, h*d) // (B, Tn, C)
	return a.O.Forward(y)
}
