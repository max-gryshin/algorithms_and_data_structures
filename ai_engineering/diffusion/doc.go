// Package diffusion is the code of lecture L15 ("Beyond text"): generative models
// for continuous data, trained on 2-D toy datasets so that every number can be
// printed, plotted and checked by hand.
//
//   - data.go     — deterministic toy datasets (two moons, spiral) with class labels;
//   - schedule.go — DDPM noise schedules (linear, cosine), ᾱ_t and closed-form noising q(x_t | x_0);
//   - model.go    — an MLP denoiser on our autograd with a sinusoidal time embedding and
//     an optional class input (with a "null" class for classifier-free guidance);
//   - ddpm.go     — DDPM training (predict the noise ε) and ancestral sampling;
//   - flow.go     — flow matching / rectified flow (predict the velocity x₁ − x₀) and Euler sampling;
//   - metrics.go  — nearest-neighbour distance between point clouds, moments;
//   - vae.go      — closed-form KL between Gaussians and the reparameterization trick;
//   - clip.go     — a CLIP-style symmetric contrastive (InfoNCE) loss on toy paired vectors;
//   - csv.go      — CSV export of samples for plotting.
//
// Points are stored as flat row-major []float32 of length 2·N: (x₀, y₀, x₁, y₁, …),
// which is exactly the data of an (N, 2) tensor.
package diffusion
