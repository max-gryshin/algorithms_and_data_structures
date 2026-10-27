package diffusion

import (
	"fmt"
	"math"
	"math/rand/v2"
)

// Schedule is a DDPM noise schedule with steps t = 1..T. Slices are indexed by t,
// element 0 is the clean data (β₀ = 0, α₀ = ᾱ₀ = 1), so formulas read exactly as in the paper:
//
//	α_t = 1 − β_t,   ᾱ_t = α₁·α₂·…·α_t,
//	q(x_t | x₀) = N(√ᾱ_t · x₀, (1 − ᾱ_t)·I).
type Schedule struct {
	T        int
	Beta     []float64 // β_t: variance of the noise added at step t
	Alpha    []float64 // α_t = 1 − β_t
	AlphaBar []float64 // ᾱ_t: how much of the original signal's variance survives after t steps
}

func newSchedule(betas []float64) *Schedule {
	s := &Schedule{T: len(betas) - 1, Beta: betas,
		Alpha: make([]float64, len(betas)), AlphaBar: make([]float64, len(betas))}
	prod := 1.0
	for t, b := range betas {
		if b < 0 || b >= 1 {
			panic(fmt.Sprintf("diffusion: β_%d = %g out of [0, 1)", t, b))
		}
		s.Alpha[t] = 1 - b
		prod *= 1 - b
		s.AlphaBar[t] = prod
	}
	return s
}

// LinearSchedule spaces β linearly from beta1 (t = 1) to betaT (t = T) — the schedule of
// Ho et al. 2020 (T = 1000, β from 1e-4 to 0.02). For 2-D toys T ≈ 100 with a larger βT suffices.
func LinearSchedule(T int, beta1, betaT float64) *Schedule {
	b := make([]float64, T+1)
	for t := 1; t <= T; t++ {
		if T == 1 {
			b[t] = beta1
			continue
		}
		b[t] = beta1 + (betaT-beta1)*float64(t-1)/float64(T-1)
	}
	return newSchedule(b)
}

// CosineSchedule is the schedule of Nichol & Dhariwal 2021: ᾱ_t follows
// f(t) = cos²((t/T + s)/(1 + s) · π/2) / f(0), and β_t = 1 − ᾱ_t/ᾱ_{t−1}, clipped at 0.999.
// It destroys information more evenly than the linear schedule.
func CosineSchedule(T int, s float64) *Schedule {
	f := func(t int) float64 {
		c := math.Cos((float64(t)/float64(T) + s) / (1 + s) * math.Pi / 2)
		return c * c
	}
	b := make([]float64, T+1)
	for t := 1; t <= T; t++ {
		b[t] = min(1-f(t)/f(t-1), 0.999)
	}
	return newSchedule(b)
}

// SNR returns the signal-to-noise ratio ᾱ_t/(1 − ᾱ_t) at step t (t ≥ 1).
func (s *Schedule) SNR(t int) float64 { return s.AlphaBar[t] / (1 - s.AlphaBar[t]) }

// Noise applies the closed form x_t = √ᾱ_t·x₀ + √(1 − ᾱ_t)·ε to one coordinate.
func (s *Schedule) Noise(x0 float64, t int, eps float64) float64 {
	return math.Sqrt(s.AlphaBar[t])*x0 + math.Sqrt(1-s.AlphaBar[t])*eps
}

// NoiseIterative runs the forward chain x_k = √α_k·x_{k−1} + √β_k·ε_k for k = 1..t —
// the definition from which the closed form is derived. Used by tests and the lecture.
func (s *Schedule) NoiseIterative(rng *rand.Rand, x0 float64, t int) float64 {
	x := x0
	for k := 1; k <= t; k++ {
		x = math.Sqrt(s.Alpha[k])*x + math.Sqrt(s.Beta[k])*rng.NormFloat64()
	}
	return x
}
