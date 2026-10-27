// Package prob implements the probability and information-theory toolkit of the course
// (lecture L2): numerically stable softmax / log-softmax / log-sum-exp, temperature,
// entropy, cross-entropy, KL divergence, perplexity, categorical sampling
// and a character n-gram language model with add-k smoothing.
//
// Functions are generic over float32 (model data: logits, probabilities)
// and float64; all sums are accumulated in float64. Logarithms are natural
// (results in nats); use NatsToBits to convert.
package prob

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Float is the set of element types accepted by the package.
type Float interface{ ~float32 | ~float64 }

// NatsToBits converts an information quantity from nats (ln) to bits (log2).
func NatsToBits(nats float64) float64 { return nats / math.Ln2 }

// Max returns the largest element of x (-Inf for an empty slice).
func Max[F Float](x []F) float64 {
	m := math.Inf(-1)
	for _, v := range x {
		m = max(m, float64(v))
	}
	return m
}

// Argmax returns the index of the largest element (the first one on ties), or -1 for an empty slice.
func Argmax[F Float](x []F) int {
	best := -1
	for i, v := range x {
		if best < 0 || v > x[best] {
			best = i
		}
	}
	return best
}

// LogSumExp returns log(Σ exp(x_i)) computed as m + log(Σ exp(x_i − m)), m = max(x).
// No exp overflows because every exponent is ≤ 0. Returns -Inf for an empty slice.
func LogSumExp[F Float](x []F) float64 {
	m := Max(x)
	if math.IsInf(m, 0) {
		return m // all -Inf (or empty) → -Inf; any +Inf → +Inf
	}
	var s float64
	for _, v := range x {
		s += math.Exp(float64(v) - m)
	}
	return m + math.Log(s)
}

// SoftmaxNaive computes exp(z_i) / Σ exp(z_j) literally. It overflows for logits ≳ 89 (float32)
// or ≳ 710 (float64) and exists only to demonstrate why Softmax subtracts the max.
func SoftmaxNaive[F Float](logits []F) []F {
	out := make([]F, len(logits))
	var sum F
	for i, z := range logits {
		out[i] = F(math.Exp(float64(z)))
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}

// Softmax turns logits into a probability distribution: p_i = exp(z_i − m) / Σ_j exp(z_j − m),
// m = max(z). Subtracting m does not change the result but makes it overflow-free.
func Softmax[F Float](logits []F) []F {
	if len(logits) == 0 {
		return nil
	}
	m := Max(logits)
	e := make([]float64, len(logits))
	var sum float64
	for i, z := range logits {
		e[i] = math.Exp(float64(z) - m)
		sum += e[i]
	}
	out := make([]F, len(logits))
	for i := range e {
		out[i] = F(e[i] / sum)
	}
	return out
}

// LogSoftmax returns log p_i = z_i − LogSumExp(z). It stays finite even when p_i underflows to 0.
func LogSoftmax[F Float](logits []F) []F {
	lse := LogSumExp(logits)
	out := make([]F, len(logits))
	for i, z := range logits {
		out[i] = F(float64(z) - lse)
	}
	return out
}

// SoftmaxTemperature returns Softmax(z / T). T < 1 sharpens the distribution, T > 1 flattens it,
// T → ∞ gives the uniform distribution. T <= 0 is treated as the T → 0 limit: greedy one-hot at argmax.
func SoftmaxTemperature[F Float](logits []F, temperature float64) []F {
	if temperature <= 0 {
		out := make([]F, len(logits))
		if i := Argmax(logits); i >= 0 {
			out[i] = 1
		}
		return out
	}
	scaled := make([]float64, len(logits))
	for i, z := range logits {
		scaled[i] = float64(z) / temperature
	}
	p := Softmax(scaled)
	out := make([]F, len(p))
	for i, v := range p {
		out[i] = F(v)
	}
	return out
}

// Entropy returns H(p) = −Σ p_i ln p_i in nats (0·ln 0 is taken as 0).
func Entropy[F Float](p []F) float64 {
	var h float64
	for _, v := range p {
		if v > 0 {
			h -= float64(v) * math.Log(float64(v))
		}
	}
	return h
}

// CrossEntropy returns H(p, q) = −Σ p_i ln q_i in nats: the average code length when data come
// from p but are encoded with a code optimal for q. +Inf if q_i = 0 where p_i > 0.
func CrossEntropy[F Float](p, q []F) float64 {
	mustSameLen(len(p), len(q))
	var h float64
	for i, pi := range p {
		if pi > 0 {
			h -= float64(pi) * math.Log(float64(q[i]))
		}
	}
	return h
}

// KL returns the Kullback–Leibler divergence D_KL(p‖q) = Σ p_i ln(p_i / q_i) = H(p, q) − H(p) ≥ 0.
// It is not symmetric: KL(p, q) != KL(q, p) in general.
func KL[F Float](p, q []F) float64 {
	mustSameLen(len(p), len(q))
	var d float64
	for i, pi := range p {
		if pi > 0 {
			d += float64(pi) * (math.Log(float64(pi)) - math.Log(float64(q[i])))
		}
	}
	return d
}

// CrossEntropyLogits is the training loss for one position: −log softmax(z)[target]
// = LogSumExp(z) − z_target. Never computes softmax explicitly, so it never takes log(0).
func CrossEntropyLogits[F Float](logits []F, target int) float64 {
	return LogSumExp(logits) - float64(logits[target])
}

// Perplexity converts a mean negative log-likelihood (nats per token) into exp(meanNLL):
// "the model is as unsure as if it chose uniformly among this many options".
func Perplexity(meanNLL float64) float64 { return math.Exp(meanNLL) }

// BitsPerByte converts a total negative log-likelihood (nats) of a text into bits per UTF-8 byte.
// Unlike per-token perplexity it is comparable across tokenizers.
func BitsPerByte(totalNLL float64, nBytes int) float64 {
	return NatsToBits(totalNLL) / float64(nBytes)
}

// Sample draws an index i with probability probs[i] / Σ probs (inverse-CDF method, O(n)):
// draw u ~ U[0, Σ), return the first i whose cumulative sum exceeds u.
func Sample[F Float](rng *rand.Rand, probs []F) int {
	var total float64
	for _, p := range probs {
		total += float64(p)
	}
	u := rng.Float64() * total
	var c float64
	last := -1
	for i, p := range probs {
		if p <= 0 {
			continue
		}
		c += float64(p)
		last = i
		if u < c {
			return i
		}
	}
	return last // rounding: u landed at the very end
}

// CDF is a precomputed cumulative distribution for repeated sampling in O(log n) by binary search.
type CDF struct{ cum []float64 }

// NewCDF builds the cumulative sums of probs (they need not be normalised).
func NewCDF[F Float](probs []F) *CDF {
	cum := make([]float64, len(probs))
	var c float64
	for i, p := range probs {
		c += float64(p)
		cum[i] = c
	}
	return &CDF{cum: cum}
}

// Sample draws one index: the first i with cum[i] > u, u ~ U[0, total).
func (c *CDF) Sample(rng *rand.Rand) int {
	n := len(c.cum)
	u := rng.Float64() * c.cum[n-1]
	i := sort.Search(n, func(i int) bool { return c.cum[i] > u })
	return min(i, n-1)
}

func mustSameLen(a, b int) {
	if a != b {
		panic("prob: distributions of different length")
	}
}
