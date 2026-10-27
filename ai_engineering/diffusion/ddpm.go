package diffusion

import (
	"math"
	"math/rand/v2"
)

// TrainDDPM trains d to predict the noise (Ho et al. 2020, Algorithm 1):
//
//	x₀ ~ data,  t ~ U{1..T},  ε ~ N(0, I),  x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε,
//	loss = ‖ε − ε_θ(x_t, t)‖².
//
// The network sees τ = t/T. Returns the loss per step.
func TrainDDPM(d *Denoiser, s *Schedule, data *Dataset, cfg TrainConfig, rng *rand.Rand) []float64 {
	return train(d, data, cfg, rng, func(rng *rand.Rand, x0 []float32) ([]float32, []float64, []float32) {
		b := len(x0) / 2
		xt := make([]float32, len(x0))
		eps := make([]float32, len(x0))
		tau := make([]float64, b)
		for i := range b {
			t := 1 + rng.IntN(s.T)
			tau[i] = float64(t) / float64(s.T)
			for c := range 2 {
				e := rng.NormFloat64()
				eps[2*i+c] = float32(e)
				xt[2*i+c] = float32(s.Noise(float64(x0[2*i+c]), t, e))
			}
		}
		return xt, tau, eps
	})
}

// PosteriorStd is σ_t = √β̃_t, β̃_t = β_t·(1 − ᾱ_{t−1})/(1 − ᾱ_t): the std of the true
// reverse step q(x_{t−1} | x_t, x₀). Ho et al. found σ_t² = β_t to work about as well.
func (s *Schedule) PosteriorStd(t int) float64 {
	return math.Sqrt(s.Beta[t] * (1 - s.AlphaBar[t-1]) / (1 - s.AlphaBar[t]))
}

// Timesteps returns steps+1 decreasing times from T to 0, evenly spaced
// (steps = T gives T, T−1, …, 0). Sampling on a sub-sequence is "respacing".
func (s *Schedule) Timesteps(steps int) []int {
	steps = max(1, min(steps, s.T))
	ts := make([]int, steps+1)
	for i := range ts {
		ts[i] = int(math.Round(float64(s.T) * float64(steps-i) / float64(steps)))
	}
	return ts
}

// SampleDDPM draws n points by ancestral sampling (Algorithm 2), from x_T ~ N(0, I) down to x₀.
// With steps = T every step is
//
//	x_{t−1} = 1/√α_t · (x_t − β_t/√(1−ᾱ_t) · ε_θ(x_t, t)) + σ_t·z,   z ~ N(0, I), no noise at the last step,
//
// σ_t = PosteriorStd(t). With steps < T the same formula is applied between times t > t′
// of Timesteps(steps) with α = ᾱ_t/ᾱ_{t′} and β = 1 − α (respacing, Nichol & Dhariwal 2021).
// label is a class id or NoClass; guidance is the classifier-free guidance weight w
// (1 = plain conditional). snap (may be nil) receives x after each step with its time.
func SampleDDPM(d *Denoiser, s *Schedule, n, steps, label int, guidance float64, rng *rand.Rand, snap Snapshot) []float32 {
	return sample(d, s, n, steps, label, guidance, rng, snap, false)
}

// SampleDDIM is the deterministic sampler of Song et al. 2020 (η = 0) for the same
// trained ε-model: estimate the clean point, then re-noise it to the earlier time
// with the *predicted* noise instead of fresh noise:
//
//	x̂₀ = (x_t − √(1−ᾱ_t)·ε_θ)/√ᾱ_t,   x_{t′} = √ᾱ_{t′}·x̂₀ + √(1−ᾱ_{t′})·ε_θ.
//
// It tolerates far fewer steps than ancestral sampling.
func SampleDDIM(d *Denoiser, s *Schedule, n, steps, label int, guidance float64, rng *rand.Rand, snap Snapshot) []float32 {
	return sample(d, s, n, steps, label, guidance, rng, snap, true)
}

func sample(d *Denoiser, s *Schedule, n, steps, label int, guidance float64, rng *rand.Rand, snap Snapshot, ddim bool) []float32 {
	x := Gaussian(rng, n)
	labels := classLabels(d, n, label)
	tau := make([]float64, n)
	ts := s.Timesteps(steps)
	if snap != nil {
		snap(ts[0], x)
	}
	for k := 0; k+1 < len(ts); k++ {
		t, tp := ts[k], ts[k+1]
		for i := range tau {
			tau[i] = float64(t) / float64(s.T)
		}
		eps := d.predict(x, tau, labels, guidance)
		ab, abp := s.AlphaBar[t], s.AlphaBar[tp]
		if ddim {
			for i := range x {
				e := float64(eps[i])
				x0 := (float64(x[i]) - math.Sqrt(1-ab)*e) / math.Sqrt(ab)
				x[i] = float32(math.Sqrt(abp)*x0 + math.Sqrt(1-abp)*e)
			}
		} else {
			alpha := ab / abp
			beta := 1 - alpha
			k := beta / math.Sqrt(1-ab)
			inv := 1 / math.Sqrt(alpha)
			sigma := 0.0
			if tp > 0 {
				sigma = math.Sqrt(beta * (1 - abp) / (1 - ab))
			}
			for i := range x {
				m := inv * (float64(x[i]) - k*float64(eps[i]))
				if sigma > 0 {
					m += sigma * rng.NormFloat64()
				}
				x[i] = float32(m)
			}
		}
		if snap != nil {
			snap(tp, x)
		}
	}
	return x
}
