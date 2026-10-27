package diffusion

import "math/rand/v2"

// TrainFlow trains d as a velocity field by (conditional) flow matching with the
// straight-line path of rectified flow (Lipman et al. 2022, Liu et al. 2022):
//
//	x₀ ~ N(0, I) (noise),  x₁ ~ data,  t ~ U[0, 1],
//	x_t = (1 − t)·x₀ + t·x₁,   target v = dx_t/dt = x₁ − x₀,
//	loss = ‖v − v_θ(x_t, t)‖².
//
// Note the convention: here t = 0 is noise and t = 1 is data (the opposite of DDPM).
func TrainFlow(d *Denoiser, data *Dataset, cfg TrainConfig, rng *rand.Rand) []float64 {
	return train(d, data, cfg, rng, func(rng *rand.Rand, x1 []float32) ([]float32, []float64, []float32) {
		b := len(x1) / 2
		xt := make([]float32, len(x1))
		v := make([]float32, len(x1))
		tau := make([]float64, b)
		for i := range b {
			t := rng.Float64()
			tau[i] = t
			for c := range 2 {
				x0 := rng.NormFloat64()
				x1c := float64(x1[2*i+c])
				xt[2*i+c] = float32((1-t)*x0 + t*x1c)
				v[2*i+c] = float32(x1c - x0)
			}
		}
		return xt, tau, v
	})
}

// SampleFlow integrates dx/dt = v_θ(x, t) from t = 0 (noise) to t = 1 with `steps`
// Euler steps: x ← x + Δt·v_θ(x, t), Δt = 1/steps. The straighter the learned paths,
// the fewer steps are needed. label / guidance / snap as in SampleDDPM.
func SampleFlow(d *Denoiser, n, steps, label int, guidance float64, rng *rand.Rand, snap Snapshot) []float32 {
	x := Gaussian(rng, n)
	labels := classLabels(d, n, label)
	tau := make([]float64, n)
	dt := 1 / float64(steps)
	if snap != nil {
		snap(0, x)
	}
	for k := range steps {
		for i := range tau {
			tau[i] = float64(k) * dt
		}
		v := d.predict(x, tau, labels, guidance)
		for i := range x {
			x[i] += float32(dt) * v[i]
		}
		if snap != nil {
			snap(k+1, x)
		}
	}
	return x
}
