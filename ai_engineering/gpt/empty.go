package gpt

import (
	"algorithms_and_data_structures/ai_engineering/nn"
)

// NewEmpty builds a model of the given architecture with zero weights (norm gains = 1)
// and no random initialization. Loaders of pretrained weights (L12) use it: drawing
// 124M Gaussian numbers for GPT-2 only to overwrite them takes seconds, NewEmpty takes
// the time of the allocation. Every parameter has the shape New would give it, so
// NamedParameters lists exactly the same names and shapes.
func NewEmpty(cfg Config) *Model {
	cfg = cfg.withDefaults()
	if err := cfg.Validate(); err != nil {
		panic(err)
	}
	d, hd := cfg.DModel, cfg.HeadDim()
	linear := func(in, out int, bias bool) *nn.Linear {
		l := &nn.Linear{W: nn.Zeros("linear.W", in, out)}
		if bias {
			l.B = nn.Zeros("linear.b", out)
		}
		return l
	}
	m := &Model{Cfg: cfg, Tok: &nn.Embedding{W: nn.Zeros("tok", cfg.Vocab, d)}}
	switch cfg.Pos {
	case PosLearned:
		m.Pos = &nn.Embedding{W: nn.Zeros("pos", cfg.Context, d)}
	case PosSinusoidal:
		m.Sin = nn.SinusoidalPE(cfg.Context, d)
	case PosRoPE:
		m.Rope = nn.NewRoPE(hd, cfg.Context, cfg.RoPETheta)
	}
	for range cfg.Layers {
		b := &nn.Block{
			Norm1: nn.NewNorm(cfg.Norm, d, cfg.Bias, cfg.NormEps),
			Norm2: nn.NewNorm(cfg.Norm, d, cfg.Bias, cfg.NormEps),
			Attn: &nn.CausalSelfAttention{
				Q:     linear(d, cfg.Heads*hd, cfg.Bias),
				K:     linear(d, cfg.KVHeads*hd, cfg.Bias),
				V:     linear(d, cfg.KVHeads*hd, cfg.Bias),
				O:     linear(cfg.Heads*hd, d, cfg.Bias),
				Heads: cfg.Heads, KVHeads: cfg.KVHeads, HeadDim: hd, Rope: m.Rope,
			},
		}
		if cfg.FFN == nn.FFNSwiGLU {
			b.FFN = &nn.SwiGLU{Gate: linear(d, cfg.FFHidden, cfg.Bias), Up: linear(d, cfg.FFHidden, cfg.Bias),
				Down: linear(cfg.FFHidden, d, cfg.Bias)}
		} else {
			b.FFN = &nn.FFN{Up: linear(d, cfg.FFHidden, cfg.Bias), Down: linear(cfg.FFHidden, d, cfg.Bias)}
		}
		m.Blocks = append(m.Blocks, b)
	}
	m.NormF = nn.NewNorm(cfg.Norm, d, cfg.Bias, cfg.NormEps)
	if !cfg.Tied {
		m.Head = &nn.Linear{W: nn.Zeros("head", d, cfg.Vocab)}
	}
	return m
}
