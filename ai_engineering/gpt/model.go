package gpt

import (
	"fmt"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/nn"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Model is a decoder-only transformer:
//
//	ids (B, T) → Tok (+ Pos) → (B, T, d) → L × Block → NormF → LM head → logits (B, T, V)
type Model struct {
	Cfg    Config
	Tok    *nn.Embedding  // (V, d)
	Pos    *nn.Embedding  // (Context, d), only for PosLearned
	Sin    *tensor.Tensor // (Context, d), only for PosSinusoidal
	Rope   *nn.RoPE       // shared by all layers, only for PosRoPE
	Blocks []*nn.Block
	NormF  nn.Layer   // final norm before the LM head
	Head   *nn.Linear // (d, V) without bias; nil when Cfg.Tied (the head is Tokᵀ)

	rng *rand.Rand // non-nil in training mode: source of dropout masks
}

// New builds a model with GPT-2-style initialization (every matrix N(0, InitStd²),
// residual projections scaled by 1/√(2L), norms with gain 1 and bias 0).
func New(rng *rand.Rand, cfg Config) *Model {
	cfg = cfg.withDefaults()
	if err := cfg.Validate(); err != nil {
		panic(err)
	}
	m := &Model{Cfg: cfg, Tok: nn.NewEmbedding(rng, cfg.Vocab, cfg.DModel, cfg.InitStd)}
	m.Tok.W.Name = "tok"
	switch cfg.Pos {
	case PosLearned:
		m.Pos = nn.NewEmbedding(rng, cfg.Context, cfg.DModel, cfg.InitStd)
		m.Pos.W.Name = "pos"
	case PosSinusoidal:
		m.Sin = nn.SinusoidalPE(cfg.Context, cfg.DModel)
	case PosRoPE:
		m.Rope = nn.NewRoPE(cfg.HeadDim(), cfg.Context, cfg.RoPETheta)
	}
	bc := cfg.blockConfig()
	for range cfg.Layers {
		m.Blocks = append(m.Blocks, nn.NewBlock(rng, bc, m.Rope))
	}
	m.NormF = nn.NewNorm(cfg.Norm, cfg.DModel, cfg.Bias, cfg.NormEps)
	if !cfg.Tied {
		m.Head = nn.NewLinearStd(rng, cfg.DModel, cfg.Vocab, false, cfg.InitStd)
		m.Head.W.Name = "head"
	}
	return m
}

// Forward returns logits (B, T, V) for a batch of equal-length sequences ids (B, T).
func (m *Model) Forward(ids [][]int) *autograd.Var { return m.ForwardAt(ids, 0) }

// ForwardAt is Forward for tokens that sit at absolute positions pos..pos+T−1
// (pos shifts learned/sinusoidal position embeddings and the RoPE angles).
func (m *Model) ForwardAt(ids [][]int, pos int) *autograd.Var {
	x := m.embed(ids, pos) // (B, T, d)
	if m.rng != nil && m.Cfg.Dropout > 0 {
		x = autograd.Dropout(x, m.Cfg.Dropout, m.rng) // training mode only
	}
	for _, b := range m.Blocks {
		x = b.Forward(x, pos) // (B, T, d)
	}
	return m.head(x)
}

// embed maps ids (B, T) at positions pos… to the residual stream (B, T, d).
func (m *Model) embed(ids [][]int, pos int) *autograd.Var {
	if len(ids) == 0 || len(ids[0]) == 0 {
		panic("gpt: empty batch")
	}
	t := len(ids[0])
	if pos+t > m.Cfg.Context {
		panic(fmt.Sprintf("gpt: positions up to %d exceed context %d", pos+t, m.Cfg.Context))
	}
	x := m.Tok.ForwardBatch(ids) // (B, T, d)
	switch m.Cfg.Pos {
	case PosLearned:
		x = autograd.Add(x, m.Pos.Forward(nn.Positions(pos, t))) // + (T, d), broadcast over B
	case PosSinusoidal:
		x = autograd.Add(x, autograd.NewVar(nn.Rows(m.Sin, pos, pos+t))) // + (T, d), constant
	}
	return x
}

// head applies the final norm and the LM head: (B, T, d) → logits (B, T, V).
func (m *Model) head(x *autograd.Var) *autograd.Var {
	x = m.NormF.Forward(x) // (B, T, d)
	if m.Head != nil {
		return m.Head.Forward(x) // (B, T, d)·(d, V) = (B, T, V)
	}
	return autograd.MatMulBT(x, m.Tok.W) // (B, T, d)·(V, d)ᵀ: tied head without a transposed copy
}

// Train switches the model to training mode: dropout (Cfg.Dropout) draws its masks
// from rng. Eval switches it off again. Without dropout both are no-ops.
func (m *Model) Train(rng *rand.Rand) {
	m.rng = rng
	for _, b := range m.Blocks {
		b.Dropout, b.Rng = m.Cfg.Dropout, rng
	}
}

// Eval switches to inference mode (no dropout).
func (m *Model) Eval() { m.Train(nil) }

// Loss is the mean next-token cross-entropy. targets has the shape of ids;
// a target < 0 is ignored (padding, prompt tokens in SFT).
func (m *Model) Loss(ids, targets [][]int) *autograd.Var {
	flat := make([]int, 0, len(targets)*len(targets[0]))
	for _, row := range targets {
		flat = append(flat, row...)
	}
	return autograd.CrossEntropy(m.Forward(ids), flat) // logits (B, T, V) vs B·T targets
}

func (m *Model) Parameters() []*autograd.Var {
	ps := []*autograd.Var{m.Tok.W}
	if m.Pos != nil {
		ps = append(ps, m.Pos.W)
	}
	for _, b := range m.Blocks {
		ps = append(ps, b.Parameters()...)
	}
	ps = append(ps, m.NormF.Parameters()...)
	if m.Head != nil {
		ps = append(ps, m.Head.W)
	}
	return ps
}

// NumParams counts the parameters of the built model (tied weights counted once).
func (m *Model) NumParams() int { return nn.CountParams(m.Parameters()) }
