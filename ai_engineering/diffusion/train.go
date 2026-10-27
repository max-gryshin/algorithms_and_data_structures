package diffusion

import (
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/optim"
)

// TrainConfig holds the hyper-parameters shared by DDPM and flow-matching training.
type TrainConfig struct {
	Steps int     // optimizer steps
	Batch int     // points per step
	LR    float64 // peak learning rate (Adam, warmup + cosine decay to LR/10)
	// CondDrop is the probability of replacing a label with NoClass during training
	// (10–20 % in practice). It only matters for class-conditional models.
	CondDrop float64
}

// batchFn turns clean points x₀ (B×2) into the network input and its regression target.
type batchFn func(rng *rand.Rand, x0 []float32) (xt []float32, tau []float64, target []float32)

// train is the loop common to DDPM and flow matching: sample a batch of data,
// corrupt it, regress the target with MSE. Returns the loss of every step.
func train(d *Denoiser, data *Dataset, cfg TrainConfig, rng *rand.Rand, mk batchFn) []float64 {
	params := d.Parameters()
	opt := optim.NewAdam(params, cfg.LR)
	sched := optim.CosineSchedule{MaxLR: cfg.LR, MinLR: cfg.LR / 10, WarmupSteps: max(1, cfg.Steps/50), TotalSteps: cfg.Steps}
	losses := make([]float64, 0, cfg.Steps)
	x0 := make([]float32, 2*cfg.Batch)
	var labels []int
	if d.NumClasses > 0 {
		labels = make([]int, cfg.Batch)
	}
	for step := range cfg.Steps {
		for i := range cfg.Batch {
			j := rng.IntN(data.Len())
			x0[2*i], x0[2*i+1] = data.Point(j)
			if labels != nil {
				labels[i] = data.Labels[j]
				if rng.Float64() < cfg.CondDrop {
					labels[i] = NoClass
				}
			}
		}
		xt, tau, target := mk(rng, x0)
		opt.ZeroGrad()
		loss := mse(d.Forward(xt, tau, labels), target)
		loss.Backward()
		optim.ClipGradNorm(params, 1.0)
		opt.SetLR(sched.LR(step))
		opt.Step()
		losses = append(losses, float64(loss.Item()))
	}
	return losses
}

// classLabels returns nil for an unconditional request, else n copies of label.
func classLabels(d *Denoiser, n, label int) []int {
	if d.NumClasses == 0 || label == NoClass {
		return nil
	}
	l := make([]int, n)
	for i := range l {
		l[i] = label
	}
	return l
}

// Snapshot, if non-nil, is called during sampling with the current points; DDPM
// passes the step t (T..0), flow matching the Euler step k (0..N). Used to export
// the whole denoising trajectory to CSV.
type Snapshot func(step int, x []float32)
