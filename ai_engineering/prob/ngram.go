package prob

import (
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
)

// BOS pads the context at the beginning of a text ("beginning of sequence").
// It may appear in contexts but is never predicted.
const BOS = '\x02'

// NGram is a character n-gram language model with add-k (Laplace for k = 1) smoothing:
//
//	p(x | ctx) = (count(ctx, x) + k) / (count(ctx) + k·|V|),  ctx = previous n−1 characters.
//
// n = 1 is a unigram model (no context), n = 2 a bigram model, etc.
type NGram struct {
	N     int     // order: context length is N−1
	K     float64 // add-k smoothing constant (0 = pure maximum likelihood)
	vocab []rune
	index map[rune]int
	// counts[ctx][j] = how many times vocab[j] followed the context string ctx.
	counts map[string][]uint32
	totals map[string]uint32
}

// Vocab returns the sorted set of distinct characters of text.
func Vocab(text string) []rune {
	set := map[rune]struct{}{}
	for _, r := range text {
		set[r] = struct{}{}
	}
	v := make([]rune, 0, len(set))
	for r := range set {
		v = append(v, r)
	}
	slices.Sort(v)
	return v
}

// NewNGram returns an empty model of order n over the given vocabulary.
func NewNGram(n int, k float64, vocab []rune) *NGram {
	if n < 1 {
		panic(fmt.Sprintf("prob: NGram order %d < 1", n))
	}
	m := &NGram{N: n, K: k, vocab: slices.Clone(vocab), index: make(map[rune]int, len(vocab)),
		counts: map[string][]uint32{}, totals: map[string]uint32{}}
	for i, r := range m.vocab {
		m.index[r] = i
	}
	return m
}

// Vocab returns the model vocabulary (a copy).
func (m *NGram) Vocab() []rune { return slices.Clone(m.vocab) }

// Train adds the n-gram counts of text. Characters outside the vocabulary panic:
// build the vocabulary from all data you will train and evaluate on.
func (m *NGram) Train(text string) {
	ctx := m.startContext()
	for _, r := range text {
		j, ok := m.index[r]
		if !ok {
			panic(fmt.Sprintf("prob: character %q is not in the vocabulary", r))
		}
		key := string(ctx)
		row := m.counts[key]
		if row == nil {
			row = make([]uint32, len(m.vocab))
			m.counts[key] = row
		}
		row[j]++
		m.totals[key]++
		ctx = m.shift(ctx, r)
	}
}

// Prob returns p(next | context), using only the last N−1 characters of context
// (shorter contexts are left-padded with BOS). Unknown next characters get probability 0.
// An unseen context with K = 0 falls back to the uniform distribution.
func (m *NGram) Prob(context string, next rune) float64 {
	j, ok := m.index[next]
	if !ok {
		return 0
	}
	key := string(m.contextOf(context))
	return m.prob(key, j)
}

func (m *NGram) prob(key string, j int) float64 {
	total := float64(m.totals[key])
	denom := total + m.K*float64(len(m.vocab))
	if denom == 0 {
		return 1 / float64(len(m.vocab))
	}
	var c float64
	if row := m.counts[key]; row != nil {
		c = float64(row[j])
	}
	return (c + m.K) / denom
}

// Dist returns the full next-character distribution p(· | context), aligned with Vocab().
func (m *NGram) Dist(context string) []float64 {
	key := string(m.contextOf(context))
	p := make([]float64, len(m.vocab))
	for j := range p {
		p[j] = m.prob(key, j)
	}
	return p
}

// Generate continues prompt with length sampled characters. Temperature reshapes each
// distribution as softmax(log p / T): T = 1 samples the model as is, T → 0 is greedy.
func (m *NGram) Generate(rng *rand.Rand, prompt string, length int, temperature float64) string {
	var b strings.Builder
	b.WriteString(prompt)
	ctx := m.contextOf(prompt)
	logp := make([]float64, len(m.vocab))
	for range length {
		p := m.Dist(string(ctx))
		if temperature != 1 {
			for j, v := range p {
				logp[j] = math.Log(v) // log 0 = -Inf → stays impossible after softmax
			}
			p = SoftmaxTemperature(logp, temperature)
		}
		r := m.vocab[Sample(rng, p)]
		b.WriteRune(r)
		ctx = m.shift(ctx, r)
	}
	return b.String()
}

// NLL returns the total negative log-likelihood (nats) of text under the model and the number
// of predicted characters. The first characters are predicted from BOS-padded contexts.
func (m *NGram) NLL(text string) (total float64, count int) {
	ctx := m.startContext()
	for _, r := range text {
		p := m.Prob(string(ctx), r)
		total -= math.Log(p) // p = 0 → +Inf: an unsmoothed model is "infinitely surprised"
		count++
		ctx = m.shift(ctx, r)
	}
	return total, count
}

// Perplexity returns exp(mean NLL per character) of text under the model.
func (m *NGram) Perplexity(text string) float64 {
	total, n := m.NLL(text)
	return Perplexity(total / float64(n))
}

func (m *NGram) startContext() []rune {
	ctx := make([]rune, m.N-1)
	for i := range ctx {
		ctx[i] = BOS
	}
	return ctx
}

// contextOf returns the last N−1 runes of s, left-padded with BOS.
func (m *NGram) contextOf(s string) []rune {
	ctx := m.startContext()
	for _, r := range s {
		ctx = m.shift(ctx, r)
	}
	return ctx
}

// shift drops the oldest character of the context and appends r.
func (m *NGram) shift(ctx []rune, r rune) []rune {
	if len(ctx) == 0 {
		return ctx
	}
	copy(ctx, ctx[1:])
	ctx[len(ctx)-1] = r
	return ctx
}
