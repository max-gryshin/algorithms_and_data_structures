package diffusion

import (
	"math"
	"math/rand/v2"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/nn"
	"algorithms_and_data_structures/ai_engineering/optim"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// L2Normalize scales every row of x (..., d) to unit length: x / √(Σx² + 1e-12).
// After it, a dot product is the cosine similarity.
func L2Normalize(x *autograd.Var) *autograd.Var {
	sq := autograd.SumLast(autograd.Pow(x, 2))
	inv := autograd.Pow(autograd.Add(sq, autograd.Const(1e-12)), -0.5)
	return autograd.Mul(x, inv)
}

// ContrastiveLoss is the symmetric InfoNCE loss of CLIP for a batch of N matching pairs
// (row i of img goes with row i of txt), both (N, d):
//
//	S = norm(img)·norm(txt)ᵀ / τ                       (N, N) cosine similarities / temperature
//	loss = ½·(CE(S, [0..N−1]) + CE(Sᵀ, [0..N−1]))      image→text and text→image classification
//
// Every row is an N-way classification "which text is mine?", the other N − 1 texts
// in the batch being the negatives.
func ContrastiveLoss(img, txt *autograd.Var, temperature float32) *autograd.Var {
	s := Similarity(img, txt, temperature)
	n := s.Shape()[0]
	diag := make([]int, n)
	for i := range diag {
		diag[i] = i
	}
	li := autograd.CrossEntropy(s, diag)
	lt := autograd.CrossEntropy(autograd.Transpose(s, 0, 1), diag)
	return autograd.Scale(autograd.Add(li, lt), 0.5)
}

// Similarity returns the (N, M) matrix of cosine similarities divided by temperature.
func Similarity(img, txt *autograd.Var, temperature float32) *autograd.Var {
	a, b := L2Normalize(img), L2Normalize(txt)
	return autograd.Scale(autograd.MatMul(a, autograd.Transpose(b, 0, 1)), 1/temperature)
}

// ToyPairs imitates an (image, caption) dataset: each pair shares a hidden "concept"
// z ~ N(0, I_k), and the two modalities see it through different random linear maps
// plus their own noise: img = z·A + n₁ (dImg features), txt = z·B + n₂ (dTxt features).
// Nothing in img and txt is directly comparable — a shared space must be learned.
type ToyPairs struct {
	N, DImg, DTxt int
	Img, Txt      []float32 // N×DImg and N×DTxt, row-major
}

// MakeToyPairs generates n pairs from a k-dimensional concept space.
func MakeToyPairs(rng *rand.Rand, n, k, dImg, dTxt int, noise float64) *ToyPairs {
	a := randMat(rng, k, dImg)
	b := randMat(rng, k, dTxt)
	p := &ToyPairs{N: n, DImg: dImg, DTxt: dTxt, Img: make([]float32, n*dImg), Txt: make([]float32, n*dTxt)}
	z := make([]float64, k)
	for i := range n {
		for j := range z {
			z[j] = rng.NormFloat64()
		}
		project(z, a, dImg, p.Img[i*dImg:(i+1)*dImg], noise, rng)
		project(z, b, dTxt, p.Txt[i*dTxt:(i+1)*dTxt], noise, rng)
	}
	return p
}

func randMat(rng *rand.Rand, r, c int) []float64 {
	m := make([]float64, r*c)
	for i := range m {
		m[i] = rng.NormFloat64() / math.Sqrt(float64(r))
	}
	return m
}

func project(z, m []float64, c int, out []float32, noise float64, rng *rand.Rand) {
	for j := range c {
		var s float64
		for i, zi := range z {
			s += zi * m[i*c+j]
		}
		out[j] = float32(s + noise*rng.NormFloat64())
	}
}

// CLIP is two linear encoders into a shared embedding space with a fixed temperature
// (real CLIP: a ViT / ResNet and a text transformer, and a learned temperature).
type CLIP struct {
	ImgProj, TxtProj *nn.Linear
	Temperature      float32
}

// NewCLIP builds encoders dImg→dim and dTxt→dim.
func NewCLIP(rng *rand.Rand, dImg, dTxt, dim int, temperature float32) *CLIP {
	return &CLIP{
		ImgProj:     nn.NewLinear(rng, dImg, dim, false, 1),
		TxtProj:     nn.NewLinear(rng, dTxt, dim, false, 1),
		Temperature: temperature,
	}
}

// Parameters implements nn.Module.
func (c *CLIP) Parameters() []*autograd.Var { return nn.Collect(c.ImgProj, c.TxtProj) }

// Encode embeds rows idx of the pairs: returns (len(idx), dim) image and text embeddings.
func (c *CLIP) Encode(p *ToyPairs, idx []int) (img, txt *autograd.Var) {
	xi := make([]float32, 0, len(idx)*p.DImg)
	xt := make([]float32, 0, len(idx)*p.DTxt)
	for _, i := range idx {
		xi = append(xi, p.Img[i*p.DImg:(i+1)*p.DImg]...)
		xt = append(xt, p.Txt[i*p.DTxt:(i+1)*p.DTxt]...)
	}
	img = c.ImgProj.Forward(autograd.NewVar(tensor.FromSlice(xi, len(idx), p.DImg)))
	txt = c.TxtProj.Forward(autograd.NewVar(tensor.FromSlice(xt, len(idx), p.DTxt)))
	return img, txt
}

// TrainCLIP runs Adam on random batches of pairs and returns the loss per step.
func TrainCLIP(c *CLIP, p *ToyPairs, steps, batch int, lr float64, rng *rand.Rand) []float64 {
	opt := optim.NewAdam(c.Parameters(), lr)
	losses := make([]float64, 0, steps)
	for range steps {
		idx := rng.Perm(p.N)[:batch]
		img, txt := c.Encode(p, idx)
		opt.ZeroGrad()
		loss := ContrastiveLoss(img, txt, c.Temperature)
		loss.Backward()
		opt.Step()
		losses = append(losses, float64(loss.Item()))
	}
	return losses
}

// RetrievalAccuracy is the fraction of rows i of the similarity matrix whose largest
// entry is on the diagonal: "for image i, the best-matching text is its own caption".
func RetrievalAccuracy(sim *autograd.Var) float64 {
	sh := sim.Shape()
	n, m := sh[0], sh[1]
	d := sim.Data.Data()
	hit := 0
	for i := range n {
		best := 0
		for j := 1; j < m; j++ {
			if d[i*m+j] > d[i*m+best] {
				best = j
			}
		}
		if best == i {
			hit++
		}
	}
	return float64(hit) / float64(n)
}
