// Package gpt assembles a decoder-only transformer (lecture L8) from the blocks in
// package nn. One config-driven Model covers both families:
//
//	GPT-2: learned absolute positions, LayerNorm, GELU MLP (4·d), biases, tied LM head
//	Llama: RoPE, RMSNorm, SwiGLU MLP (≈ 8/3·d), no biases, GQA, untied LM head
//
// L9 adds dropout, named parameters, replicas for data-parallel training and checkpoints
// (training itself lives in gpt/train); L10 adds the KV-cache (Session) used by gpt/generate.
package gpt

import (
	"fmt"
	"math"

	"algorithms_and_data_structures/ai_engineering/nn"
)

// PosKind selects how token positions enter the model (L6).
type PosKind int

const (
	PosLearned    PosKind = iota // table (Context, d) added to token embeddings (GPT-2)
	PosSinusoidal                // fixed sin/cos table added to token embeddings (Vaswani et al.)
	PosRoPE                      // rotation of q and k inside every attention layer (Llama)
	PosNone                      // no positions at all (NoPE) — for experiments
)

func (p PosKind) String() string {
	return [...]string{"learned", "sinusoidal", "rope", "none"}[p]
}

// Config describes the architecture. Zero values of KVHeads and InitStd get defaults (see Validate).
type Config struct {
	Name     string
	Vocab    int // V
	Context  int // maximum sequence length (n_ctx / block_size)
	DModel   int // d: width of the residual stream
	Layers   int // L
	Heads    int // h
	KVHeads  int // h_kv: = h for MHA, 1 for MQA, between for GQA
	FFHidden int // hidden width of the MLP: 4·d (GPT-2) or ≈ 8/3·d rounded (Llama)

	Pos  PosKind
	Norm nn.NormKind
	FFN  nn.FFNKind
	Bias bool // biases in every Linear and LayerNorm (GPT-2)
	Tied bool // LM head = token embedding matrixᵀ (weight tying, L6)

	NormEps   float32
	RoPETheta float64 // RoPE base θ: 10 000 (Llama 2), 500 000 (Llama 3)
	InitStd   float32 // std of every weight matrix at init (GPT-2: 0.02)

	// Dropout (L9) is the probability of zeroing an activation in training mode: after
	// the embeddings and on the output of every attention and MLP sublayer. 0 = off.
	Dropout float32
}

// GPT2Small is GPT-2 124M (Radford et al., 2019; Hugging Face "gpt2").
var GPT2Small = Config{Name: "GPT-2 124M", Vocab: 50257, Context: 1024, DModel: 768, Layers: 12, Heads: 12,
	KVHeads: 12, FFHidden: 3072, Pos: PosLearned, Norm: nn.NormLayer, FFN: nn.FFNGELU, Bias: true, Tied: true,
	NormEps: 1e-5, InitStd: 0.02}

// Llama2_7B is Llama 2 7B (Touvron et al., 2023): MHA, 32K SentencePiece vocabulary.
var Llama2_7B = Config{Name: "Llama-2-7B", Vocab: 32000, Context: 4096, DModel: 4096, Layers: 32, Heads: 32,
	KVHeads: 32, FFHidden: 11008, Pos: PosRoPE, Norm: nn.NormRMS, FFN: nn.FFNSwiGLU,
	NormEps: 1e-5, RoPETheta: 10000, InitStd: 0.02}

// Llama3_8B is Llama 3 8B (Meta, 2024): GQA with 8 K/V heads, 128K tiktoken-based vocabulary.
var Llama3_8B = Config{Name: "Llama-3-8B", Vocab: 128256, Context: 8192, DModel: 4096, Layers: 32, Heads: 32,
	KVHeads: 8, FFHidden: 14336, Pos: PosRoPE, Norm: nn.NormRMS, FFN: nn.FFNSwiGLU,
	NormEps: 1e-5, RoPETheta: 500000, InitStd: 0.02}

// TinyGPT2 and TinyLlama are CPU-sized models of both families for tests and L9.
func TinyGPT2(vocab, context int) Config {
	return Config{Name: "tiny-gpt2", Vocab: vocab, Context: context, DModel: 32, Layers: 2, Heads: 4, KVHeads: 4,
		FFHidden: 128, Pos: PosLearned, Norm: nn.NormLayer, FFN: nn.FFNGELU, Bias: true, Tied: true,
		NormEps: 1e-5, InitStd: 0.02}
}

func TinyLlama(vocab, context int) Config {
	return Config{Name: "tiny-llama", Vocab: vocab, Context: context, DModel: 32, Layers: 2, Heads: 4, KVHeads: 2,
		FFHidden: SwiGLUHidden(32, 8), Pos: PosRoPE, Norm: nn.NormRMS, FFN: nn.FFNSwiGLU,
		NormEps: 1e-5, RoPETheta: 10000, InitStd: 0.02}
}

// SwiGLUHidden is the Llama rule for the MLP width: 2/3 of 4·d (so that three matrices
// cost as much as the two of a 4·d MLP), rounded up to a multiple of `multiple`.
// d = 4096, multiple = 256 → 10 922.67 → 11 008 (Llama 2 7B).
func SwiGLUHidden(d, multiple int) int {
	h := 8 * d / 3
	return (h + multiple - 1) / multiple * multiple
}

// HeadDim is d_head = d / h.
func (c Config) HeadDim() int { return c.DModel / c.Heads }

// withDefaults fills KVHeads, NormEps and InitStd when left zero.
func (c Config) withDefaults() Config {
	if c.KVHeads == 0 {
		c.KVHeads = c.Heads
	}
	if c.NormEps == 0 {
		c.NormEps = 1e-5
	}
	if c.InitStd == 0 {
		c.InitStd = 0.02
	}
	if c.RoPETheta == 0 {
		c.RoPETheta = 10000
	}
	return c
}

// Validate reports an inconsistent config.
func (c Config) Validate() error {
	c = c.withDefaults()
	switch {
	case c.Vocab <= 0 || c.Context <= 0 || c.DModel <= 0 || c.Layers <= 0 || c.Heads <= 0 || c.FFHidden <= 0:
		return fmt.Errorf("gpt: %s: sizes must be positive", c.Name)
	case c.DModel%c.Heads != 0:
		return fmt.Errorf("gpt: %s: d=%d not divisible by h=%d", c.Name, c.DModel, c.Heads)
	case c.Heads%c.KVHeads != 0:
		return fmt.Errorf("gpt: %s: h=%d not divisible by h_kv=%d", c.Name, c.Heads, c.KVHeads)
	case c.Pos == PosRoPE && c.HeadDim()%2 != 0:
		return fmt.Errorf("gpt: %s: RoPE needs an even head dim, got %d", c.Name, c.HeadDim())
	case c.Pos == PosSinusoidal && c.DModel%2 != 0:
		return fmt.Errorf("gpt: %s: sinusoidal PE needs an even d", c.Name)
	}
	return nil
}

// blockConfig translates the model config into nn.BlockConfig. GPT-2 scales the init of
// the two projections that write into the residual stream (W_O and W_down) by 1/√(2L):
// 2L sublayers add to the stream, so without it the stream's variance grows with depth.
func (c Config) blockConfig() nn.BlockConfig {
	out := c.InitStd / float32(math.Sqrt(2*float64(c.Layers)))
	return nn.BlockConfig{
		Attention: nn.AttentionConfig{DModel: c.DModel, Heads: c.Heads, KVHeads: c.KVHeads, Bias: c.Bias,
			Std: c.InitStd, OutStd: out},
		FFHidden: c.FFHidden, FFN: c.FFN, Norm: c.Norm, NormEps: c.NormEps,
	}
}
