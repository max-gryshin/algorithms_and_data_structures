package evals

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Mean returns the arithmetic mean (0 for an empty slice).
func Mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// StdErr is the standard error of the mean: s / √n, with the sample standard
// deviation s computed with n − 1 in the denominator.
func StdErr(xs []float64) float64 {
	n := float64(len(xs))
	if n < 2 {
		return 0
	}
	m := Mean(xs)
	var ss float64
	for _, x := range xs {
		ss += (x - m) * (x - m)
	}
	return math.Sqrt(ss/(n-1)) / math.Sqrt(n)
}

// NormalCI is the textbook 95 % interval mean ± 1.96·SE.
func NormalCI(xs []float64) (lo, hi float64) {
	m, se := Mean(xs), StdErr(xs)
	return m - 1.96*se, m + 1.96*se
}

// Bootstrap estimates a (1 − alpha) confidence interval of the mean by resampling:
// draw len(xs) values with replacement, take the mean, repeat iters times, and
// return the alpha/2 and 1 − alpha/2 percentiles of those means.
func Bootstrap(xs []float64, iters int, alpha float64, rng *rand.Rand) (lo, hi float64) {
	if len(xs) == 0 {
		return 0, 0
	}
	means := make([]float64, iters)
	for i := range means {
		var s float64
		for range xs {
			s += xs[rng.IntN(len(xs))]
		}
		means[i] = s / float64(len(xs))
	}
	return Percentile(means, alpha/2*100), Percentile(means, (1-alpha/2)*100)
}

// Percentile returns the p-th percentile (0..100) with linear interpolation between
// closest ranks (NumPy's default): rank = p/100·(n−1).
func Percentile(xs []float64, p float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := append([]float64(nil), xs...)
	sort.Float64s(s)
	rank := p / 100 * float64(len(s)-1)
	lo := int(math.Floor(rank))
	hi := min(lo+1, len(s)-1)
	return s[lo] + (rank-float64(lo))*(s[hi]-s[lo])
}

// McNemar compares two systems on the same cases using only discordant pairs:
// b = cases A passes and B fails, c = cases B passes and A fails. Under H0
// ("equally good") each discordant case is a fair coin, so the exact two-sided
// p-value is 2·P(X ≤ min(b, c)) for X ~ Binomial(b + c, ½), capped at 1.
func McNemar(b, c int) float64 {
	n := b + c
	if n == 0 {
		return 1
	}
	k := min(b, c)
	// P(X ≤ k) = Σ_{i=0..k} C(n, i) / 2ⁿ, computed in log space to avoid overflow.
	var p float64
	for i := 0; i <= k; i++ {
		p += math.Exp(logChoose(n, i) - float64(n)*math.Ln2)
	}
	return math.Min(1, 2*p)
}

func logChoose(n, k int) float64 {
	a, _ := math.Lgamma(float64(n + 1))
	b, _ := math.Lgamma(float64(k + 1))
	c, _ := math.Lgamma(float64(n - k + 1))
	return a - b - c
}

// SampleSize returns how many cases are needed so that the 95 % interval of an
// accuracy near p has half-width margin: n = 1.96²·p(1−p)/margin².
func SampleSize(p, margin float64) int {
	return int(math.Ceil(1.96 * 1.96 * p * (1 - p) / (margin * margin)))
}
