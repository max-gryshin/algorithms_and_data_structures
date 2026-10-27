package gpt

import (
	"fmt"
	"strings"

	"algorithms_and_data_structures/ai_engineering/nn"
)

// Back-of-the-envelope calculator for lecture L8: parameters, FLOPs and memory
// derived from the config alone (no model is built, so it works for 8B models).

// ParamCount is the exact parameter count split by component.
type ParamCount struct {
	TokenEmb  int64 // V·d
	PosEmb    int64 // Context·d for learned positions, else 0
	Attention int64 // per layer: W_Q, W_K, W_V, W_O (+ biases)
	MLP       int64 // per layer: 2 or 3 matrices (+ biases)
	Norms     int64 // per layer: two norms
	FinalNorm int64
	Head      int64 // d·V when untied, 0 when tied
	Total     int64
}

// PerLayer is the parameter count of one block.
func (p ParamCount) PerLayer() int64 { return p.Attention + p.MLP + p.Norms }

// NonEmbedding is N in Kaplan et al. (2020): everything except embeddings and the LM head.
func (p ParamCount) NonEmbedding() int64 { return p.Total - p.TokenEmb - p.PosEmb - p.Head }

// Params counts parameters exactly as Model.NumParams would for the same config.
func (c Config) Params() ParamCount {
	c = c.withDefaults()
	d, v := int64(c.DModel), int64(c.Vocab)
	kv := int64(c.KVHeads * c.HeadDim())
	f := int64(c.FFHidden)
	bias := func(n int64) int64 {
		if c.Bias {
			return n
		}
		return 0
	}
	norm := d // gain
	if c.Norm == nn.NormLayer {
		norm += bias(d) // + shift
	}
	p := ParamCount{TokenEmb: v * d}
	if c.Pos == PosLearned {
		p.PosEmb = int64(c.Context) * d
	}
	p.Attention = d*d + bias(d) + 2*(d*kv+bias(kv)) + d*d + bias(d) // Q, K, V, O
	if c.FFN == nn.FFNSwiGLU {
		p.MLP = 2*(d*f+bias(f)) + f*d + bias(d) // gate, up, down
	} else {
		p.MLP = d*f + bias(f) + f*d + bias(d) // up, down
	}
	p.Norms = 2 * norm
	p.FinalNorm = norm
	if !c.Tied {
		p.Head = d * v
	}
	p.Total = p.TokenEmb + p.PosEmb + int64(c.Layers)*p.PerLayer() + p.FinalNorm + p.Head
	return p
}

// ApproxParams is the textbook estimate 12·L·d² + V·d (MHA, 4·d GELU MLP, tied head;
// biases, norms and positions ignored). L8 shows where 12 comes from: 4d² attention + 8d² MLP.
func (c Config) ApproxParams() int64 {
	d := int64(c.DModel)
	return 12*int64(c.Layers)*d*d + int64(c.Vocab)*d
}

// ForwardFLOPsPerToken is the cost of one token's forward pass in a context of ctx tokens:
// 2 FLOPs (multiply + add) per weight of every matmul, i.e. 2·(N_nonemb + V·d) for the
// projections and the LM head, plus attention scores and mixing: Q·Kᵀ and A·V each cost
// 2·ctx·d_attn per layer. The embedding lookup is a copy and costs nothing.
func (c Config) ForwardFLOPsPerToken(ctx int) float64 {
	p := c.Params()
	matmul := 2 * float64(p.NonEmbedding()+int64(c.Vocab)*int64(c.DModel))
	attn := 4 * float64(c.Layers) * float64(ctx) * float64(c.DModel)
	return matmul + attn
}

// TrainingFLOPs is the classic C ≈ 6·N·D: forward 2N, backward 4N per token
// (the backward pass computes gradients for both the inputs and the weights of each matmul).
func (c Config) TrainingFLOPs(tokens float64) float64 { return 6 * float64(c.Params().Total) * tokens }

// Memory holds byte estimates.
type Memory struct {
	WeightsBF16   float64 // inference: 2 bytes/param
	TrainingState float64 // mixed precision + Adam: 2 (bf16 weights) + 2 (grads) + 4 (fp32 master) + 8 (m, v) = 16 bytes/param
	KVPerToken    float64 // KV-cache: 2 (K, V) · L · h_kv · d_head · 2 bytes (bf16)
	Activations   float64 // training activations for one sequence of Context tokens, see ActivationBytes
}

// Memory estimates memory for inference and training.
func (c Config) Memory() Memory {
	c = c.withDefaults()
	n := float64(c.Params().Total)
	return Memory{
		WeightsBF16:   2 * n,
		TrainingState: 16 * n,
		KVPerToken:    float64(2 * c.Layers * c.KVHeads * c.HeadDim() * 2),
		Activations:   c.ActivationBytes(1, c.Context),
	}
}

// ActivationBytes is the activation memory of training without recomputation, from
// Korthikanti et al. (2022) for a GPT block with 16-bit activations and dropout:
//
//	per layer: s·b·d·(34 + 5·h·s/d) bytes
//
// s = sequence length, b = batch, h = heads. The 5·h·s² term is the (B, h, T, T) attention
// matrices (softmax output, dropout mask, ...). For Llama-style blocks it is a rough estimate.
func (c Config) ActivationBytes(batch, seq int) float64 {
	s, b, d, h := float64(seq), float64(batch), float64(c.DModel), float64(c.Heads)
	return float64(c.Layers) * s * b * d * (34 + 5*h*s/d)
}

// Report formats the calculator output for one config (used by cmd/gptcalc).
func (c Config) Report(trainTokens float64) string {
	c = c.withDefaults()
	p := c.Params()
	m := c.Memory()
	var sb strings.Builder
	fmt.Fprintf(&sb, "== %s ==\n", c.Name)
	fmt.Fprintf(&sb, "V=%d ctx=%d d=%d L=%d h=%d h_kv=%d d_head=%d ffn=%d pos=%s\n",
		c.Vocab, c.Context, c.DModel, c.Layers, c.Heads, c.KVHeads, c.HeadDim(), c.FFHidden, c.Pos)
	fmt.Fprintf(&sb, "params total        %15d  (%.2fB)\n", p.Total, float64(p.Total)/1e9)
	fmt.Fprintf(&sb, "  token embedding   %15d\n", p.TokenEmb)
	fmt.Fprintf(&sb, "  position emb.     %15d\n", p.PosEmb)
	fmt.Fprintf(&sb, "  per layer         %15d  (attn %d, mlp %d, norms %d)\n", p.PerLayer(), p.Attention, p.MLP, p.Norms)
	fmt.Fprintf(&sb, "  all layers        %15d\n", int64(c.Layers)*p.PerLayer())
	fmt.Fprintf(&sb, "  final norm        %15d\n", p.FinalNorm)
	fmt.Fprintf(&sb, "  LM head           %15d\n", p.Head)
	fmt.Fprintf(&sb, "  non-embedding     %15d\n", p.NonEmbedding())
	fmt.Fprintf(&sb, "  12·L·d² + V·d     %15d  (%.1f%% of exact)\n", c.ApproxParams(), 100*float64(c.ApproxParams())/float64(p.Total))
	fmt.Fprintf(&sb, "forward FLOPs/token %15.3g  (2N = %.3g; ctx = %d)\n", c.ForwardFLOPsPerToken(c.Context), 2*float64(p.Total), c.Context)
	fmt.Fprintf(&sb, "training FLOPs      %15.3g  (6·N·D, D = %.3g tokens)\n", c.TrainingFLOPs(trainTokens), trainTokens)
	fmt.Fprintf(&sb, "weights bf16        %15s\n", bytes(m.WeightsBF16))
	fmt.Fprintf(&sb, "train state (16B/p) %15s\n", bytes(m.TrainingState))
	fmt.Fprintf(&sb, "activations, 1 seq  %15s  (no recomputation, ctx = %d)\n", bytes(m.Activations), c.Context)
	fmt.Fprintf(&sb, "KV-cache / token    %15s  (full ctx: %s)\n", bytes(m.KVPerToken), bytes(m.KVPerToken*float64(c.Context)))
	return sb.String()
}

func bytes(b float64) string {
	switch {
	case b >= 1<<30:
		return fmt.Sprintf("%.2f GiB", b/(1<<30))
	case b >= 1<<20:
		return fmt.Sprintf("%.2f MiB", b/(1<<20))
	case b >= 1<<10:
		return fmt.Sprintf("%.2f KiB", b/(1<<10))
	}
	return fmt.Sprintf("%.0f B", b)
}
