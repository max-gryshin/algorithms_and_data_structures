package nn

import (
	"fmt"
	"math"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Transformer building blocks (lectures L7–L8). Activations are (B, T, C).

// Layer is a module that maps a tensor to a tensor of the same leading shape.
type Layer interface {
	Module
	Forward(x *autograd.Var) *autograd.Var
}

// ---------- normalization ----------

// LayerNorm: y = g ⊙ (x − μ)/√(σ² + eps) + b over the last axis (GPT-2).
type LayerNorm struct {
	G, B *autograd.Var // (C); B is nil without bias
	Eps  float32
}

func NewLayerNorm(dim int, bias bool, eps float32) *LayerNorm {
	n := &LayerNorm{G: Ones("ln.g", dim), Eps: eps}
	if bias {
		n.B = Zeros("ln.b", dim)
	}
	return n
}

func (n *LayerNorm) Forward(x *autograd.Var) *autograd.Var {
	return autograd.LayerNorm(x, n.G, n.B, n.Eps)
}

func (n *LayerNorm) Parameters() []*autograd.Var {
	if n.B == nil {
		return []*autograd.Var{n.G}
	}
	return []*autograd.Var{n.G, n.B}
}

// RMSNorm: y = g ⊙ x / √(mean(x²) + eps) over the last axis (Llama).
type RMSNorm struct {
	G   *autograd.Var // (C)
	Eps float32
}

func NewRMSNorm(dim int, eps float32) *RMSNorm { return &RMSNorm{G: Ones("rms.g", dim), Eps: eps} }

func (n *RMSNorm) Forward(x *autograd.Var) *autograd.Var { return autograd.RMSNorm(x, n.G, n.Eps) }

func (n *RMSNorm) Parameters() []*autograd.Var { return []*autograd.Var{n.G} }

// Ones returns a parameter filled with 1 (the initial gain of a norm).
func Ones(name string, shape ...int) *autograd.Var {
	t := tensor.New(shape...)
	for i := range t.Data() {
		t.Data()[i] = 1
	}
	p := autograd.Param(t)
	p.Name = name
	return p
}

// ---------- attention ----------

// CausalMask returns the additive mask (tq, tk): 0 where query i may look at key j,
// −∞ where it may not. Queries are the last tq of the tk positions (bottom-right
// alignment), so query i sits at position tk−tq+i and sees keys j ≤ tk−tq+i.
// Without a KV-cache tq == tk and this is the usual lower-triangular mask; with a
// cache (L10) the new tokens see all cached keys.
func CausalMask(tq, tk int) *tensor.Tensor {
	m := tensor.New(tq, tk)
	d := m.Data()
	inf := float32(math.Inf(-1))
	for i := range tq {
		for j := tk - tq + i + 1; j < tk; j++ {
			d[i*tk+j] = inf
		}
	}
	return m
}

// AttentionConfig sizes a CausalSelfAttention layer.
type AttentionConfig struct {
	DModel  int  // C
	Heads   int  // h: query heads, d_head = C / h
	KVHeads int  // h_kv: key/value heads; h for MHA, 1 for MQA, in between for GQA (must divide h)
	Bias    bool // biases in Q, K, V, O projections (GPT-2: yes, Llama: no)
	Std     float32
	OutStd  float32 // std of W_O (GPT-2 scales it by 1/√(2L))
}

// CausalSelfAttention is multi-head attention with optional grouped K/V heads:
//
//	Q = x·W_Q, K = x·W_K, V = x·W_V   (split into heads)
//	A = softmax(Q·Kᵀ/√d_head + mask),  y = concat_heads(A·V)·W_O
type CausalSelfAttention struct {
	Q, K, V, O *Linear
	Heads      int
	KVHeads    int
	HeadDim    int
	Rope       *RoPE // nil: no rotary embedding

	// Weights holds the attention probabilities of the last Forward, (B, h, T_q, T_k).
	// Read it to look at what heads attend to (induction heads, attention sinks — L7).
	Weights *tensor.Tensor
}

func NewCausalSelfAttention(rng *rand.Rand, c AttentionConfig, rope *RoPE) *CausalSelfAttention {
	if c.KVHeads == 0 {
		c.KVHeads = c.Heads
	}
	if c.DModel%c.Heads != 0 || c.Heads%c.KVHeads != 0 {
		panic(fmt.Sprintf("nn: attention d=%d h=%d h_kv=%d: need h | d and h_kv | h", c.DModel, c.Heads, c.KVHeads))
	}
	hd := c.DModel / c.Heads
	if rope != nil && rope.HeadDim != hd {
		panic(fmt.Sprintf("nn: RoPE head dim %d, attention head dim %d", rope.HeadDim, hd))
	}
	return &CausalSelfAttention{
		Q:     NewLinearStd(rng, c.DModel, c.Heads*hd, c.Bias, c.Std),
		K:     NewLinearStd(rng, c.DModel, c.KVHeads*hd, c.Bias, c.Std),
		V:     NewLinearStd(rng, c.DModel, c.KVHeads*hd, c.Bias, c.Std),
		O:     NewLinearStd(rng, c.Heads*hd, c.DModel, c.Bias, c.OutStd),
		Heads: c.Heads, KVHeads: c.KVHeads, HeadDim: hd, Rope: rope,
	}
}

// Forward maps x: (B, T, C) → (B, T, C). pos is the absolute position of x's first
// token: 0 for a full sequence; with a KV-cache (L10) it is the number of cached tokens.
func (a *CausalSelfAttention) Forward(x *autograd.Var, pos int) *autograd.Var {
	s := x.Shape()
	b, t := s[0], s[1]
	h, kv, d := a.Heads, a.KVHeads, a.HeadDim

	split := func(y *autograd.Var, heads int) *autograd.Var {
		y = autograd.Reshape(y, b, t, heads, d) // (B, T, heads, d)
		return autograd.Transpose(y, 1, 2)      // (B, heads, T, d)
	}
	q := split(a.Q.Forward(x), h)  // (B, h, T, d)
	k := split(a.K.Forward(x), kv) // (B, kv, T, d)
	v := split(a.V.Forward(x), kv) // (B, kv, T, d)
	if a.Rope != nil {
		q = a.Rope.Apply(q, pos)
		k = a.Rope.Apply(k, pos)
	}
	// Fused attention (autograd.CausalAttention): S = Q·Kᵀ/√d_head, causal mask,
	// softmax, ·V — in one pass per query row. GQA without copying K/V: query head j
	// reads K/V head j / g. The composite version with the same math is in L7 and in
	// autograd's tests; the fused one is several times faster on the CPU (L9).
	y, att := autograd.CausalAttention(q, k, v, float32(1/math.Sqrt(float64(d)))) // (B, h, T, d)
	a.Weights = att                                                               // (B, h, T, T), rows sum to 1
	y = autograd.Transpose(y, 1, 2)                                               // (B, T, h, d)
	y = autograd.Reshape(y, b, t, h*d)                                            // (B, T, C): "concat heads" is a reshape
	return a.O.Forward(y)                                                         // (B, T, C)
}

func (a *CausalSelfAttention) Parameters() []*autograd.Var { return Collect(a.Q, a.K, a.V, a.O) }

// ---------- feed-forward ----------

// FFN is the GPT-2 MLP: y = GELU(x·W_up + b)·W_down + b, hidden = 4·C.
type FFN struct {
	Up, Down *Linear
}

func NewFFN(rng *rand.Rand, dim, hidden int, bias bool, std, outStd float32) *FFN {
	return &FFN{Up: NewLinearStd(rng, dim, hidden, bias, std), Down: NewLinearStd(rng, hidden, dim, bias, outStd)}
}

func (f *FFN) Forward(x *autograd.Var) *autograd.Var {
	return f.Down.Forward(autograd.GELU(f.Up.Forward(x))) // (B,T,C) → (B,T,H) → (B,T,C)
}

func (f *FFN) Parameters() []*autograd.Var { return Collect(f.Up, f.Down) }

// SwiGLU is the Llama MLP (Shazeer, 2020): y = (SiLU(x·W_gate) ⊙ (x·W_up))·W_down.
// Three matrices instead of two, so hidden ≈ 8/3·C keeps the parameter count of a 4·C FFN.
type SwiGLU struct {
	Gate, Up, Down *Linear
}

func NewSwiGLU(rng *rand.Rand, dim, hidden int, bias bool, std, outStd float32) *SwiGLU {
	return &SwiGLU{
		Gate: NewLinearStd(rng, dim, hidden, bias, std),
		Up:   NewLinearStd(rng, dim, hidden, bias, std),
		Down: NewLinearStd(rng, hidden, dim, bias, outStd),
	}
}

func (f *SwiGLU) Forward(x *autograd.Var) *autograd.Var {
	gate := autograd.SiLU(f.Gate.Forward(x))                   // (B, T, H)
	return f.Down.Forward(autograd.Mul(gate, f.Up.Forward(x))) // (B, T, H) → (B, T, C)
}

func (f *SwiGLU) Parameters() []*autograd.Var { return Collect(f.Gate, f.Up, f.Down) }

// ---------- block ----------

// NormKind selects the normalization layer.
type NormKind int

const (
	NormLayer NormKind = iota // LayerNorm (GPT-2)
	NormRMS                   // RMSNorm (Llama)
)

// FFNKind selects the feed-forward network.
type FFNKind int

const (
	FFNGELU   FFNKind = iota // two matrices + GELU (GPT-2)
	FFNSwiGLU                // three matrices + SiLU gate (Llama)
)

// BlockConfig sizes a transformer block.
type BlockConfig struct {
	Attention AttentionConfig
	FFHidden  int
	FFN       FFNKind
	Norm      NormKind
	NormEps   float32
}

// NewNorm builds the configured normalization over dim features.
func NewNorm(kind NormKind, dim int, bias bool, eps float32) Layer {
	if kind == NormRMS {
		return NewRMSNorm(dim, eps)
	}
	return NewLayerNorm(dim, bias, eps)
}

// Block is a Pre-LN transformer block:
//
//	x = x + Attn(Norm1(x))
//	x = x + FFN(Norm2(x))
//
// The residual stream x is never normalized itself — every sublayer reads a
// normalized copy and adds its result back (L8).
type Block struct {
	Norm1, Norm2 Layer
	Attn         *CausalSelfAttention
	FFN          Layer

	// Dropout (L9) zeroes each output of the two sublayers with probability Dropout
	// before it is added to the residual stream. It is active only while Rng != nil
	// (training mode); gpt.Model.Train / Eval switch it.
	Dropout float32
	Rng     *rand.Rand
}

func NewBlock(rng *rand.Rand, c BlockConfig, rope *RoPE) *Block {
	a := c.Attention
	blk := &Block{
		Norm1: NewNorm(c.Norm, a.DModel, a.Bias, c.NormEps),
		Norm2: NewNorm(c.Norm, a.DModel, a.Bias, c.NormEps),
		Attn:  NewCausalSelfAttention(rng, a, rope),
	}
	if c.FFN == FFNSwiGLU {
		blk.FFN = NewSwiGLU(rng, a.DModel, c.FFHidden, a.Bias, a.Std, a.OutStd)
	} else {
		blk.FFN = NewFFN(rng, a.DModel, c.FFHidden, a.Bias, a.Std, a.OutStd)
	}
	return blk
}

// Forward: x (B, T, C) → (B, T, C); pos as in CausalSelfAttention.Forward.
func (b *Block) Forward(x *autograd.Var, pos int) *autograd.Var {
	x = autograd.Add(x, b.drop(b.Attn.Forward(b.Norm1.Forward(x), pos)))
	return autograd.Add(x, b.drop(b.FFN.Forward(b.Norm2.Forward(x))))
}

func (b *Block) drop(y *autograd.Var) *autograd.Var {
	if b.Rng == nil || b.Dropout <= 0 {
		return y
	}
	return autograd.Dropout(y, b.Dropout, b.Rng)
}

// ForwardCached is Forward for inference with a KV-cache (L10): x holds only the new
// tokens, the layer's keys and values of all earlier tokens come from c. No dropout.
func (b *Block) ForwardCached(x *autograd.Var, c *KVCache) *autograd.Var {
	x = autograd.Add(x, b.Attn.ForwardCached(b.Norm1.Forward(x), c))
	return autograd.Add(x, b.FFN.Forward(b.Norm2.Forward(x)))
}

func (b *Block) Parameters() []*autograd.Var { return Collect(b.Norm1, b.Attn, b.Norm2, b.FFN) }
