// Package generate turns a trained gpt.Model into a text generator (lecture L10):
// samplers (greedy, temperature, top-k, top-p, min-p, repetition/frequency/presence
// penalties), streaming with UTF-8-safe detokenization and stop sequences,
// constrained decoding (logit masks from a JSON automaton) and speculative decoding.
package generate

import (
	"math"
	"math/rand/v2"
	"slices"
)

// SamplerConfig selects the decoding strategy. The zero value is greedy decoding.
type SamplerConfig struct {
	Temperature float64 // 0 = greedy (argmax); 1 = the model's distribution; <1 sharper, >1 flatter
	TopK        int     // keep the K most likely tokens; 0 = off
	TopP        float64 // nucleus: keep the smallest set with total probability ≥ P; 0 or 1 = off
	MinP        float64 // keep tokens with p ≥ MinP·p_max; 0 = off

	RepetitionPenalty float64 // CTRL-style, on every token already in the context: z>0 → z/r, z<0 → z·r; 0 or 1 = off
	FrequencyPenalty  float64 // OpenAI-style: z −= FrequencyPenalty · count(token)
	PresencePenalty   float64 // OpenAI-style: z −= PresencePenalty · [count(token) > 0]

	Seed uint64 // the random stream of the sampler; equal seeds give equal texts
}

// Sampler draws the next token from logits. Not safe for concurrent use (it owns an RNG).
type Sampler struct {
	Cfg SamplerConfig
	rng *rand.Rand
}

func NewSampler(cfg SamplerConfig) *Sampler {
	return &Sampler{Cfg: cfg, rng: rand.New(rand.NewPCG(cfg.Seed, 0x5eed))}
}

// Distribution applies the whole pipeline to logits and returns the probabilities the
// next token is drawn from:
//
//	penalties(history) → ÷ temperature → softmax → top-k → top-p → min-p → renormalize
//
// Masked logits (−∞, constrained decoding) get probability 0. Greedy (Temperature 0)
// returns a one-hot vector on the argmax. Speculative decoding needs this function
// rather than Sample: it compares the full distributions of two models.
func (s *Sampler) Distribution(logits []float32, history []int) []float64 {
	z := make([]float64, len(logits))
	for i, x := range logits {
		z[i] = float64(x)
	}
	c := s.Cfg
	ApplyPenalties(z, history, c.RepetitionPenalty, c.FrequencyPenalty, c.PresencePenalty)
	if c.Temperature <= 0 {
		p := make([]float64, len(z))
		p[argmax(z)] = 1
		return p
	}
	p := Softmax(z, c.Temperature)
	TopK(p, c.TopK)
	TopP(p, c.TopP)
	MinP(p, c.MinP)
	return p
}

// Sample draws a token id from Distribution(logits, history).
func (s *Sampler) Sample(logits []float32, history []int) int {
	return Draw(s.rng, s.Distribution(logits, history))
}

// Draw samples an index from probabilities p (they need not be normalized).
func Draw(rng *rand.Rand, p []float64) int {
	var total float64
	for _, x := range p {
		total += x
	}
	u := rng.Float64() * total
	last := 0
	for i, x := range p {
		if x <= 0 {
			continue
		}
		last = i
		if u < x {
			return i
		}
		u -= x
	}
	return last // rounding: u landed past the last bucket
}

// ApplyPenalties discourages tokens that already occur in history, in place:
//
//	repetition (CTRL, Keskar et al. 2019):  z > 0 → z / r,   z < 0 → z · r
//	frequency  (OpenAI):                    z −= α_f · count
//	presence   (OpenAI):                    z −= α_p · [count > 0]
//
// Frequency grows with every repetition, presence is a flat fee for "already said".
func ApplyPenalties(z []float64, history []int, rep, freq, pres float64) {
	if (rep == 0 || rep == 1) && freq == 0 && pres == 0 {
		return
	}
	count := map[int]int{}
	for _, id := range history {
		count[id]++
	}
	for id, n := range count {
		if id < 0 || id >= len(z) || math.IsInf(z[id], -1) {
			continue
		}
		if rep > 0 && rep != 1 {
			if z[id] > 0 {
				z[id] /= rep
			} else {
				z[id] *= rep
			}
		}
		z[id] -= freq*float64(n) + pres
	}
}

// Softmax with temperature: p_i = exp(z_i/τ) / Σ exp(z_j/τ), computed stably.
func Softmax(z []float64, temperature float64) []float64 {
	m := math.Inf(-1)
	for _, x := range z {
		m = max(m, x)
	}
	p := make([]float64, len(z))
	var s float64
	for i, x := range z {
		p[i] = math.Exp((x - m) / temperature)
		s += p[i]
	}
	for i := range p {
		p[i] /= s
	}
	return p
}

// TopK keeps the k largest probabilities (ties at the boundary are all kept) and
// renormalizes, in place. k ≤ 0 or k ≥ len(p) does nothing.
func TopK(p []float64, k int) {
	if k <= 0 || k >= len(p) {
		return
	}
	sorted := slices.Clone(p)
	slices.Sort(sorted)
	cut := sorted[len(sorted)-k] // the k-th largest
	for i := range p {
		if p[i] < cut {
			p[i] = 0
		}
	}
	normalize(p)
}

// TopP (nucleus sampling, Holtzman et al. 2019) keeps the most likely tokens until
// their total probability reaches top, drops the rest and renormalizes, in place.
// The token that crosses the threshold is kept, so at least one token survives.
func TopP(p []float64, top float64) {
	if top <= 0 || top >= 1 {
		return
	}
	idx := make([]int, len(p))
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(a, b int) int {
		switch {
		case p[a] > p[b]:
			return -1
		case p[a] < p[b]:
			return 1
		}
		return 0
	})
	var cum float64
	keep := len(idx)
	for r, i := range idx {
		cum += p[i]
		if cum >= top {
			keep = r + 1
			break
		}
	}
	for _, i := range idx[keep:] {
		p[i] = 0
	}
	normalize(p)
}

// MinP (Nguyen et al. 2024) keeps tokens with p ≥ minP · p_max and renormalizes, in
// place. The threshold scales with the model's confidence: strict when one token
// dominates, permissive when many are plausible.
func MinP(p []float64, minP float64) {
	if minP <= 0 {
		return
	}
	pmax := 0.0
	for _, x := range p {
		pmax = max(pmax, x)
	}
	for i := range p {
		if p[i] < minP*pmax {
			p[i] = 0
		}
	}
	normalize(p)
}

func normalize(p []float64) {
	var s float64
	for _, x := range p {
		s += x
	}
	if s == 0 {
		return
	}
	for i := range p {
		p[i] /= s
	}
}

// argmax returns the first index of the largest value (−∞ entries never win unless all are −∞).
func argmax(z []float64) int {
	best := 0
	for i, x := range z {
		if x > z[best] {
			best = i
		}
	}
	return best
}
