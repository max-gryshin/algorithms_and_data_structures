// Package scaling holds back-of-the-envelope calculators and toy simulations for
// lecture L11: parameter counts, training FLOPs and memory, Chinchilla scaling laws,
// MinHash/LSH deduplication, ring all-reduce and Mixture-of-Experts routing.
package scaling

import "math"

// ModelConfig describes a decoder-only transformer precisely enough to count its parameters.
type ModelConfig struct {
	Vocab    int // V: vocabulary size
	DModel   int // d_model: residual stream width
	Layers   int // L: number of transformer blocks
	Heads    int // h: query heads
	KVHeads  int // h_kv: key/value heads (== Heads for MHA, < Heads for GQA, 1 for MQA)
	FFHidden int // hidden width of the MLP (4·d for GPT-2, ~8/3·d for SwiGLU)

	Gated          bool // SwiGLU MLP: three matrices (gate, up, down) instead of two
	Bias           bool // linear layers have biases and norms are LayerNorm (weight+bias), as in GPT-2
	TiedEmbeddings bool // LM head reuses the token embedding matrix
	LearnedPos     int  // context length of a learned absolute position table (0 for RoPE)
}

// GPT2Small is GPT-2 124M (Radford et al., 2019).
var GPT2Small = ModelConfig{Vocab: 50257, DModel: 768, Layers: 12, Heads: 12, KVHeads: 12, FFHidden: 3072,
	Bias: true, TiedEmbeddings: true, LearnedPos: 1024}

// Llama2_7B is Llama 2 7B (Touvron et al., 2023).
var Llama2_7B = ModelConfig{Vocab: 32000, DModel: 4096, Layers: 32, Heads: 32, KVHeads: 32, FFHidden: 11008, Gated: true}

// Llama3_8B is Llama 3 8B: GQA with 8 KV heads and a 128K vocabulary.
var Llama3_8B = ModelConfig{Vocab: 128256, DModel: 4096, Layers: 32, Heads: 32, KVHeads: 8, FFHidden: 14336, Gated: true}

// HeadDim returns d_head = d_model / h.
func (c ModelConfig) HeadDim() int { return c.DModel / c.Heads }

func (c ModelConfig) normParams() int64 {
	if c.Bias {
		return 2 * int64(c.DModel) // LayerNorm: gain + bias
	}
	return int64(c.DModel) // RMSNorm: gain only
}

// linear counts a d_in×d_out matrix plus an optional bias vector.
func (c ModelConfig) linear(in, out int) int64 {
	n := int64(in) * int64(out)
	if c.Bias {
		n += int64(out)
	}
	return n
}

// AttentionParams counts W_Q, W_K, W_V, W_O of one block.
func (c ModelConfig) AttentionParams() int64 {
	d, kv := c.DModel, c.KVHeads*c.HeadDim()
	return c.linear(d, d) + 2*c.linear(d, kv) + c.linear(d, d)
}

// MLPParams counts the feed-forward network of one block.
func (c ModelConfig) MLPParams() int64 {
	if c.Gated {
		return 2*c.linear(c.DModel, c.FFHidden) + c.linear(c.FFHidden, c.DModel)
	}
	return c.linear(c.DModel, c.FFHidden) + c.linear(c.FFHidden, c.DModel)
}

// BlockParams counts one transformer block: two norms, attention and MLP.
func (c ModelConfig) BlockParams() int64 {
	return 2*c.normParams() + c.AttentionParams() + c.MLPParams()
}

// EmbeddingParams counts token (and learned position) embeddings plus an untied LM head.
func (c ModelConfig) EmbeddingParams() int64 {
	n := int64(c.Vocab)*int64(c.DModel) + int64(c.LearnedPos)*int64(c.DModel)
	if !c.TiedEmbeddings {
		n += int64(c.Vocab) * int64(c.DModel)
	}
	return n
}

// NonEmbeddingParams is the N used by Kaplan et al.: everything except embeddings and LM head.
func (c ModelConfig) NonEmbeddingParams() int64 {
	return int64(c.Layers)*c.BlockParams() + c.normParams()
}

// Params is the total parameter count.
func (c ModelConfig) Params() int64 { return c.NonEmbeddingParams() + c.EmbeddingParams() }

// KVCacheBytesPerToken is the memory one token occupies in the KV cache:
// 2 (K and V) · L · h_kv · d_head · bytesPerValue.
func (c ModelConfig) KVCacheBytesPerToken(bytesPerValue int) int64 {
	return 2 * int64(c.Layers) * int64(c.KVHeads) * int64(c.HeadDim()) * int64(bytesPerValue)
}

// TrainingFLOPs is the classic C ≈ 6·N·D: 2·N for the forward pass and 4·N for the backward pass, per token.
func TrainingFLOPs(params, tokens float64) float64 { return 6 * params * tokens }

// InferenceFLOPsPerToken is ≈ 2·N: one multiply and one add per weight.
func InferenceFLOPsPerToken(params float64) float64 { return 2 * params }

// TrainingMemory is the per-parameter memory of mixed-precision training with Adam, in bytes.
type TrainingMemory struct {
	Weights   float64 // bf16/fp16 working copy: 2 bytes/param
	Grads     float64 // bf16/fp16 gradients: 2 bytes/param
	Master    float64 // fp32 master weights: 4 bytes/param
	Optimizer float64 // fp32 Adam m and v: 4 + 4 bytes/param
}

// Total returns the sum in bytes.
func (m TrainingMemory) Total() float64 { return m.Weights + m.Grads + m.Master + m.Optimizer }

// MixedPrecisionAdam returns the 16-bytes-per-parameter breakdown (activations excluded).
func MixedPrecisionAdam(params float64) TrainingMemory {
	return TrainingMemory{Weights: 2 * params, Grads: 2 * params, Master: 4 * params, Optimizer: 8 * params}
}

// ZeROBytesPerGPU returns model-state memory per GPU for ZeRO stage 0..3 over n data-parallel GPUs
// (Rajbhandari et al., 2019): stage 1 shards optimizer states (12 B/param),
// stage 2 also gradients (2 B/param), stage 3 also weights (2 B/param).
func ZeROBytesPerGPU(params float64, n, stage int) float64 {
	const weights, grads, optim = 2.0, 2.0, 12.0
	nf := float64(n)
	switch stage {
	case 0:
		return params * (weights + grads + optim)
	case 1:
		return params * (weights + grads + optim/nf)
	case 2:
		return params * (weights + (grads+optim)/nf)
	case 3:
		return params * (weights + grads + optim) / nf
	}
	panic("scaling: ZeRO stage must be 0..3")
}

// Chinchilla holds the parametric loss fit L(N, D) = E + A/N^α + B/D^β.
type Chinchilla struct{ E, A, B, Alpha, Beta float64 }

// ChinchillaFit is "approach 3" from Hoffmann et al. (2022).
var ChinchillaFit = Chinchilla{E: 1.69, A: 406.4, B: 410.7, Alpha: 0.34, Beta: 0.28}

// Loss predicts the final pre-training loss (nats/token) for N parameters trained on D tokens.
func (c Chinchilla) Loss(n, d float64) float64 {
	return c.E + c.A/math.Pow(n, c.Alpha) + c.B/math.Pow(d, c.Beta)
}

// Optimal minimizes Loss under the budget C = 6·N·D analytically:
// N_opt = G·(C/6)^(β/(α+β)), D_opt = (C/6)/N_opt, G = (αA/(βB))^(1/(α+β)).
func (c Chinchilla) Optimal(flops float64) (n, d float64) {
	g := math.Pow(c.Alpha*c.A/(c.Beta*c.B), 1/(c.Alpha+c.Beta))
	n = g * math.Pow(flops/6, c.Beta/(c.Alpha+c.Beta))
	return n, flops / 6 / n
}

// ComputeOptimal20 applies the rule of thumb D ≈ 20·N: from C = 6·N·20·N it follows N = √(C/120).
func ComputeOptimal20(flops float64) (n, d float64) {
	n = math.Sqrt(flops / 120)
	return n, 20 * n
}

// GPUHours converts a FLOP budget to accelerator-hours at a given peak throughput and
// model FLOPs utilization (MFU, typically 0.3–0.5 for large training runs).
func GPUHours(flops, peakFLOPS, mfu float64) float64 {
	return flops / (peakFLOPS * mfu) / 3600
}
