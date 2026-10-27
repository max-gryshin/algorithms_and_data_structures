package posttrain

import (
	"math"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/nn"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

type nnLinear = nn.Linear

// LogSigmoid is log σ(z) computed stably as −(max(−z, 0) + log(1 + e^{−|z|})).
// The branches are picked by constant masks (sign(0) = +1), so the gradient is
// σ(−z) everywhere, including z = 0 where a ReLU-based |z| would give 0.
func LogSigmoid(z *autograd.Var) *autograd.Var {
	zd := z.Data.Data()
	neg, sign := make([]float32, len(zd)), make([]float32, len(zd))
	for i, v := range zd {
		sign[i] = 1
		if v < 0 {
			neg[i], sign[i] = -1, -1
		}
	}
	shape := z.Data.Shape()
	relu := autograd.Mul(z, autograd.NewVar(tensor.FromSlice(neg, shape...))) // max(−z, 0)
	abs := autograd.Mul(z, autograd.NewVar(tensor.FromSlice(sign, shape...))) // |z|
	soft := autograd.Log(autograd.Add(autograd.Const(1), autograd.Exp(autograd.Scale(abs, -1))))
	return autograd.Scale(autograd.Add(relu, soft), -1)
}

// BradleyTerryLoss is the reward-model loss: mean over pairs of −log σ(r_chosen − r_rejected).
func BradleyTerryLoss(chosen, rejected *autograd.Var) *autograd.Var {
	return autograd.Scale(autograd.Mean(LogSigmoid(autograd.Sub(chosen, rejected))), -1)
}

// DPOLoss is −log σ(β·[(log π(y_w) − log π_ref(y_w)) − (log π(y_l) − log π_ref(y_l))]),
// averaged over pairs. Arguments are sequence log-probabilities (one per pair); the
// reference ones are constants.
func DPOLoss(polW, polL, refW, refL *autograd.Var, beta float32) *autograd.Var {
	w := autograd.Sub(polW, refW) // implicit reward of the chosen answer / β
	l := autograd.Sub(polL, refL)
	return autograd.Scale(autograd.Mean(LogSigmoid(autograd.Scale(autograd.Sub(w, l), beta))), -1)
}

// Sigmoid in float64 for the hand calculations of the lecture.
func Sigmoid(z float64) float64 { return 1 / (1 + math.Exp(-z)) }

// DPOLoss64 is DPOLoss for one pair in float64.
func DPOLoss64(polW, polL, refW, refL, beta float64) float64 {
	return -math.Log(Sigmoid(beta * ((polW - refW) - (polL - refL))))
}

// PPOClip is the clipped surrogate of one token: min(ρ·A, clip(ρ, 1−ε, 1+ε)·A).
func PPOClip(ratio, adv, eps float64) float64 {
	c := math.Min(math.Max(ratio, 1-eps), 1+eps)
	return math.Min(ratio*adv, c*adv)
}

// GroupAdvantages is GRPO's advantage: (r_i − mean(r)) / (std(r) + 1e-6) within one
// group of answers to the same prompt. No value network is needed.
func GroupAdvantages(rewards []float64) []float64 {
	var mean, v float64
	for _, r := range rewards {
		mean += r
	}
	mean /= float64(len(rewards))
	for _, r := range rewards {
		v += (r - mean) * (r - mean)
	}
	std := math.Sqrt(v / float64(len(rewards)))
	out := make([]float64, len(rewards))
	for i, r := range rewards {
		out[i] = (r - mean) / (std + 1e-6)
	}
	return out
}

// KLk3 is the per-token KL estimate used by GRPO: ρ − log ρ − 1 with ρ = π_ref/π (≥ 0).
func KLk3(logp, refLogp float64) float64 {
	d := refLogp - logp
	return math.Exp(d) - d - 1
}
