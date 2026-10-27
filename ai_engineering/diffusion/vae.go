package diffusion

import (
	"math"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// KLGaussian1D is KL(N(μ₁, σ₁²) ‖ N(μ₂, σ₂²)) in nats, in closed form:
//
//	ln(σ₂/σ₁) + (σ₁² + (μ₁ − μ₂)²)/(2σ₂²) − ½.
func KLGaussian1D(mu1, sigma1, mu2, sigma2 float64) float64 {
	return math.Log(sigma2/sigma1) + (sigma1*sigma1+(mu1-mu2)*(mu1-mu2))/(2*sigma2*sigma2) - 0.5
}

// KLToStdNormal is the VAE regularizer KL(N(μ, diag σ²) ‖ N(0, I)) for a d-dimensional
// latent with σ² = exp(logVar): the sum over coordinates of ½·(μ² + σ² − 1 − ln σ²).
func KLToStdNormal(mu, logVar []float64) float64 {
	var s float64
	for i := range mu {
		s += 0.5 * (mu[i]*mu[i] + math.Exp(logVar[i]) - 1 - logVar[i])
	}
	return s
}

// Reparameterize returns z = μ + exp(logStd)·ε with ε supplied from outside. The
// randomness is an input constant, so z is a differentiable function of μ and logStd:
// ∂z/∂μ = 1, ∂z/∂logStd = σ·ε. Sampling z ~ N(μ, σ²) directly would have no gradient.
// mu and logStd broadcast against eps (shape of eps given by shape).
func Reparameterize(mu, logStd *autograd.Var, eps []float32, shape ...int) *autograd.Var {
	e := autograd.NewVar(tensor.FromSlice(eps, shape...))
	return autograd.Add(mu, autograd.Mul(autograd.Exp(logStd), e))
}
