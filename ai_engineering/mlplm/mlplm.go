// Package mlplm is the neural language model of Bengio et al. (2003) at the
// character level (lecture L4):
//
//	context ids (B, N) → Embedding → concat (B, N·D) → Linear+tanh (B, H) → Linear (B, V) → softmax
//
// It is the first model in the course whose token embeddings are learned.
package mlplm

import (
	"math"
	"math/rand/v2"
	"slices"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/nn"
)

// Vocab maps characters to ids 0..Size()-1 (sorted by rune for determinism).
type Vocab struct {
	runes []rune
	index map[rune]int
}

func BuildVocab(text string) *Vocab {
	set := map[rune]bool{}
	for _, r := range text {
		set[r] = true
	}
	v := &Vocab{index: map[rune]int{}}
	for r := range set {
		v.runes = append(v.runes, r)
	}
	slices.Sort(v.runes)
	for i, r := range v.runes {
		v.index[r] = i
	}
	return v
}

func (v *Vocab) Size() int { return len(v.runes) }

// Encode maps text to ids; unknown characters are skipped.
func (v *Vocab) Encode(s string) []int {
	ids := make([]int, 0, len(s))
	for _, r := range s {
		if id, ok := v.index[r]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func (v *Vocab) Decode(ids []int) string {
	rs := make([]rune, len(ids))
	for i, id := range ids {
		rs[i] = v.runes[id]
	}
	return string(rs)
}

// Config sets the model size.
type Config struct {
	Context int // N — how many previous characters the model sees
	EmbDim  int // D — embedding size per character
	Hidden  int // H — hidden layer width
}

type Model struct {
	Cfg    Config
	Vocab  int
	Emb    *nn.Embedding
	Hidden *nn.Linear
	Out    *nn.Linear
}

func NewModel(rng *rand.Rand, vocab int, cfg Config) *Model {
	m := &Model{
		Cfg:    cfg,
		Vocab:  vocab,
		Emb:    nn.NewEmbedding(rng, vocab, cfg.EmbDim, 1),
		Hidden: nn.NewLinear(rng, cfg.Context*cfg.EmbDim, cfg.Hidden, true, 5.0/3), // 5/3: tanh gain
		Out:    nn.NewLinear(rng, cfg.Hidden, vocab, true, 1),
	}
	// Small output weights make the initial distribution almost uniform: loss ≈ ln V.
	m.Out.W.Data = m.Out.W.Data.Scale(0.1)
	return m
}

func (m *Model) Parameters() []*autograd.Var { return nn.Collect(m.Emb, m.Hidden, m.Out) }

// Forward takes B contexts flattened into ctx (len B·N) and returns logits (B, V).
func (m *Model) Forward(ctx []int) *autograd.Var {
	b := len(ctx) / m.Cfg.Context
	x := m.Emb.Forward(ctx)                                // (B·N, D)
	x = autograd.Reshape(x, b, m.Cfg.Context*m.Cfg.EmbDim) // (B, N·D): concatenation is just a reshape
	h := autograd.Tanh(m.Hidden.Forward(x))                // (B, H)
	return m.Out.Forward(h)                                // (B, V)
}

// Loss is the mean cross-entropy of the next character.
func (m *Model) Loss(ctx, targets []int) *autograd.Var {
	return autograd.CrossEntropy(m.Forward(ctx), targets)
}

// Examples turns a token stream into (context, target) pairs: for every t ≥ N
// the context is ids[t−N:t] and the target is ids[t].
func Examples(ids []int, n int) (ctx [][]int, targets []int) {
	for t := n; t < len(ids); t++ {
		ctx = append(ctx, ids[t-n:t])
		targets = append(targets, ids[t])
	}
	return ctx, targets
}

// Batch samples B random examples and flattens their contexts.
func Batch(rng *rand.Rand, ctx [][]int, targets []int, b int) ([]int, []int) {
	fc := make([]int, 0, b*len(ctx[0]))
	ft := make([]int, 0, b)
	for range b {
		i := rng.IntN(len(targets))
		fc = append(fc, ctx[i]...)
		ft = append(ft, targets[i])
	}
	return fc, ft
}

// Evaluate returns the mean loss over all examples (no sampling noise).
func (m *Model) Evaluate(ctx [][]int, targets []int) float64 {
	flat := slices.Concat(ctx...)
	return float64(m.Loss(flat, targets).Item())
}

// Generate continues prompt (at least Context ids) by n sampled characters.
func (m *Model) Generate(rng *rand.Rand, prompt []int, n int, temperature float32) []int {
	out := slices.Clone(prompt)
	for range n {
		logits := m.Forward(out[len(out)-m.Cfg.Context:]).Data.Data()
		probs := make([]float64, len(logits))
		mx := slices.Max(logits)
		var s float64
		for i, z := range logits {
			probs[i] = math.Exp(float64((z - mx) / temperature))
			s += probs[i]
		}
		r := rng.Float64() * s
		next := len(probs) - 1
		for i, p := range probs {
			if r -= p; r < 0 {
				next = i
				break
			}
		}
		out = append(out, next)
	}
	return out
}

// BigramLoss is the baseline from L2: P(next | prev) from counts with add-α smoothing,
// fitted on train and evaluated (mean −ln P) on eval.
func BigramLoss(train, eval []int, vocab int, alpha float64) float64 {
	counts := make([]float64, vocab*vocab)
	rows := make([]float64, vocab)
	for i := 1; i < len(train); i++ {
		counts[train[i-1]*vocab+train[i]]++
		rows[train[i-1]]++
	}
	var nll float64
	for i := 1; i < len(eval); i++ {
		p := (counts[eval[i-1]*vocab+eval[i]] + alpha) / (rows[eval[i-1]] + alpha*float64(vocab))
		nll -= math.Log(p)
	}
	return nll / float64(len(eval)-1)
}
