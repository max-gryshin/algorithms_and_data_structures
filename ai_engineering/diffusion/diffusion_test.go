package diffusion

import (
	"bytes"
	"math"
	"math/rand/v2"
	"strings"
	"sync"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

func newRNG(seed uint64) *rand.Rand { return rand.New(rand.NewPCG(seed, 15)) }

func near(t *testing.T, name string, got, want, tol float64) {
	t.Helper()
	if math.Abs(got-want) > tol {
		t.Errorf("%s = %.6g, want %.6g ± %g", name, got, want, tol)
	}
}

// ---------- data ----------

func TestDatasetsStandardizedAndDeterministic(t *testing.T) {
	for name, gen := range map[string]func(*rand.Rand, int, float64) *Dataset{"moons": TwoMoons, "spiral": Spiral} {
		a, b := gen(newRNG(1), 1000, 0.05), gen(newRNG(1), 1000, 0.05)
		if !slicesEqual(a.X, b.X) {
			t.Fatalf("%s: same seed gave different data", name)
		}
		mean, v := Moments(a.X)
		for c := range 2 {
			near(t, name+" mean", mean[c], 0, 1e-5)
			near(t, name+" var", v[c], 1, 1e-4)
		}
		n0 := 0
		for _, l := range a.Labels {
			if l == 0 {
				n0++
			}
		}
		if n0 != 500 {
			t.Errorf("%s: %d points of class 0, want 500", name, n0)
		}
	}
}

func slicesEqual(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// ---------- schedule ----------

// The numbers of the lecture's hand-computed table (T = 100, β from 1e-4 to 0.1).
func TestLinearScheduleLectureNumbers(t *testing.T) {
	s := LinearSchedule(100, 1e-4, 0.1)
	near(t, "β₂", s.Beta[2], 0.0011090909, 1e-9)
	near(t, "ᾱ₁", s.AlphaBar[1], 0.9999, 1e-12)
	near(t, "ᾱ₂", s.AlphaBar[2], 0.99879102, 1e-8)
	near(t, "ᾱ₃", s.AlphaBar[3], 0.9966754, 1e-7)
	near(t, "ᾱ₅₀", s.AlphaBar[50], 0.28298, 1e-5)
	near(t, "ᾱ₁₀₀", s.AlphaBar[100], 0.0056188, 1e-7)
	near(t, "β₁₀₀", s.Beta[100], 0.1, 1e-12)
}

func TestAlphaBarMonotonic(t *testing.T) {
	for name, s := range map[string]*Schedule{
		"linear":     LinearSchedule(1000, 1e-4, 0.02),
		"cosine":     CosineSchedule(1000, 0.008),
		"linear-toy": LinearSchedule(100, 1e-4, 0.1),
	} {
		if s.AlphaBar[0] != 1 {
			t.Errorf("%s: ᾱ₀ = %g", name, s.AlphaBar[0])
		}
		for k := 1; k <= s.T; k++ {
			if !(s.AlphaBar[k] < s.AlphaBar[k-1]) || s.AlphaBar[k] <= 0 {
				t.Fatalf("%s: ᾱ not strictly decreasing at t=%d: %g → %g", name, k, s.AlphaBar[k-1], s.AlphaBar[k])
			}
			if k > 1 && s.SNR(k) >= s.SNR(k-1) {
				t.Fatalf("%s: SNR not decreasing at t=%d", name, k)
			}
		}
		if s.AlphaBar[s.T] > 0.01 {
			t.Errorf("%s: ᾱ_T = %g, the end is not close to pure noise", name, s.AlphaBar[s.T])
		}
	}
	// Ho et al.: T=1000 linear 1e-4..0.02 ends at ᾱ_T ≈ 4.0e-5.
	near(t, "DDPM ᾱ_1000", LinearSchedule(1000, 1e-4, 0.02).AlphaBar[1000], 4.04e-5, 0.1e-5)
}

// The closed form q(x_t | x₀) = N(√ᾱ_t·x₀, 1 − ᾱ_t) must agree, in mean and variance,
// with running the chain of t small noising steps.
func TestClosedFormMatchesIterative(t *testing.T) {
	s := LinearSchedule(100, 1e-4, 0.1)
	rng := newRNG(2)
	const x0, n = 1.5, 40000
	for _, step := range []int{1, 3, 25, 50, 100} {
		var si, si2, sc, sc2 float64
		for range n {
			a := s.NoiseIterative(rng, x0, step)
			b := s.Noise(x0, step, rng.NormFloat64())
			si, si2 = si+a, si2+a*a
			sc, sc2 = sc+b, sc2+b*b
		}
		mi, mc := si/n, sc/n
		vi, vc := si2/n-mi*mi, sc2/n-mc*mc
		wantM, wantV := math.Sqrt(s.AlphaBar[step])*x0, 1-s.AlphaBar[step]
		tolM := 4*math.Sqrt(wantV/n) + 1e-9     // 4 standard errors of the mean
		tolV := 4*wantV*math.Sqrt(2.0/n) + 1e-9 // 4 standard errors of the variance
		near(t, "iterative mean", mi, wantM, tolM)
		near(t, "closed mean", mc, wantM, tolM)
		near(t, "iterative var", vi, wantV, tolV)
		near(t, "closed var", vc, wantV, tolV)
	}
}

func TestTimestepsRespacing(t *testing.T) {
	s := LinearSchedule(100, 1e-4, 0.1)
	got := s.Timesteps(4)
	want := []int{100, 75, 50, 25, 0}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("Timesteps(4) = %v, want %v", got, want)
		}
	}
	if len(s.Timesteps(100)) != 101 || s.Timesteps(500)[1] != 99 {
		t.Fatal("Timesteps(T) must visit every step")
	}
}

func TestTimeEmbedding(t *testing.T) {
	e := TimeEmbedding(0, 8)
	for i := range 4 {
		if e[i] != 0 || e[4+i] != 1 {
			t.Fatalf("τ=0 must give sin=0, cos=1: %v", e)
		}
	}
	e = TimeEmbedding(0.001, 4) // s = 1: ω = 1, 0.01
	near(t, "sin(1)", float64(e[0]), math.Sin(1), 1e-6)
	near(t, "sin(0.01)", float64(e[1]), math.Sin(0.01), 1e-6)
	near(t, "cos(1)", float64(e[2]), math.Cos(1), 1e-6)
}

// ---------- VAE pieces ----------

func TestKLGaussianLectureNumbers(t *testing.T) {
	near(t, "KL(N(1,0.5²)‖N(0,1))", KLGaussian1D(1, 0.5, 0, 1), 0.818147, 1e-6)
	near(t, "KL(N(0,1)‖N(0,1))", KLGaussian1D(0, 1, 0, 1), 0, 1e-12)
	near(t, "KL(N(0,2²)‖N(0,1))", KLGaussian1D(0, 2, 0, 1), 0.806853, 1e-6)
	// The VAE form with log-variances agrees with the general formula, summed over coordinates.
	mu, lv := []float64{1, 0}, []float64{math.Log(0.25), math.Log(4)}
	near(t, "KLToStdNormal", KLToStdNormal(mu, lv), 0.818147+0.806853, 1e-6)
	// Monte Carlo check of the closed form: E_q[ln q(z) − ln p(z)].
	rng := newRNG(3)
	var s float64
	const n = 200000
	for range n {
		z := 1 + 0.5*rng.NormFloat64()
		lq := -math.Log(0.5) - (z-1)*(z-1)/(2*0.25)
		lp := -z * z / 2
		s += lq - lp
	}
	near(t, "KL Monte Carlo", s/n, 0.818147, 0.005)
}

// With the reparameterization z = μ + σ·ε, autograd gives an unbiased estimate of
// ∇ E[z²] = (2μ, 2σ²) w.r.t. (μ, ln σ), because E[z²] = μ² + σ².
func TestReparameterizationGradient(t *testing.T) {
	const n = 20000
	rng := newRNG(4)
	eps := make([]float32, n)
	for i := range eps {
		eps[i] = float32(rng.NormFloat64())
	}
	mu := autograd.Param(tensor.FromSlice([]float32{0.5}, 1))
	logStd := autograd.Param(tensor.FromSlice([]float32{float32(math.Log(2))}, 1))
	z := Reparameterize(mu, logStd, eps, n)
	loss := autograd.Mean(autograd.Pow(z, 2))
	loss.Backward()
	near(t, "E[z²]", float64(loss.Item()), 0.25+4, 0.1)
	near(t, "d/dμ", float64(mu.Grad.Data()[0]), 1, 0.05)
	near(t, "d/dlnσ", float64(logStd.Grad.Data()[0]), 8, 0.2)
}

// ---------- denoiser ----------

func TestDenoiserGradCheck(t *testing.T) {
	rng := newRNG(5)
	d := NewDenoiser(rng, 6, 2, 4, 2)
	x := []float32{0.3, -0.7, 1.1, 0.2, -0.5, 0.9}
	tau := []float64{0.1, 0.5, 0.9}
	target := []float32{0.1, 0.2, -0.3, 0.4, 0.5, -0.6}
	f := func() *autograd.Var { return mse(d.Forward(x, tau, []int{0, 1, NoClass}), target) }
	if rel, where := autograd.GradCheck(f, d.Parameters(), 1e-2); rel > 2e-2 {
		t.Fatalf("grad check: rel err %.3g at %s", rel, where)
	}
}

func TestGuidanceCombination(t *testing.T) {
	rng := newRNG(6)
	d := NewDenoiser(rng, 8, 1, 4, 2)
	x, tau := []float32{0.3, -0.2}, []float64{0.4}
	cond := d.predict(x, tau, []int{1}, 1)
	unc := d.predict(x, tau, []int{NoClass}, 1)
	g := d.predict(x, tau, []int{1}, 3)
	for i := range g {
		want := unc[i] + 3*(cond[i]-unc[i])
		near(t, "guided", float64(g[i]), float64(want), 1e-6)
	}
	g0 := d.predict(x, tau, []int{1}, 0)
	near(t, "w=0 is unconditional", float64(g0[0]), float64(unc[0]), 1e-6)
}

// ---------- training (shared, trained once) ----------

type trained struct {
	data            *Dataset
	sched           *Schedule
	ddpm, flow, cfg *Denoiser
	ddpmLoss        []float64
	flowLoss        []float64
}

var (
	once  sync.Once
	model trained
)

// toyModels trains small models on two moons once for all tests (~5 s without -race).
func toyModels() *trained {
	once.Do(func() {
		rng := newRNG(7)
		m := &model
		m.data = TwoMoons(rng, 2000, 0.05)
		m.sched = LinearSchedule(100, 1e-4, 0.1)
		cfg := TrainConfig{Steps: 1200, Batch: 128, LR: 3e-3, CondDrop: 0.15}
		if raceEnabled {
			cfg.Steps = 40
		}
		m.ddpm = NewDenoiser(rng, 64, 3, 16, 0)
		m.ddpmLoss = TrainDDPM(m.ddpm, m.sched, m.data, cfg, rng)
		m.flow = NewDenoiser(rng, 64, 3, 16, 0)
		m.flowLoss = TrainFlow(m.flow, m.data, cfg, rng)
		m.cfg = NewDenoiser(rng, 64, 3, 16, 2)
		TrainFlow(m.cfg, m.data, cfg, rng)
	})
	return &model
}

func mean(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

// skipQualityUnderRace runs after the code under test has been exercised.
func skipQualityUnderRace(t *testing.T) {
	if raceEnabled {
		t.Skip("quality thresholds need full training; skipped under -race")
	}
}

func TestDDPMLearnsDistribution(t *testing.T) {
	m := toyModels()
	first, last := mean(m.ddpmLoss[:min(50, len(m.ddpmLoss))]), mean(m.ddpmLoss[len(m.ddpmLoss)-min(100, len(m.ddpmLoss)):])
	rng := newRNG(8)
	noise := Gaussian(rng, 400)
	base := MeanNNDist(noise, m.data.X)
	x := SampleDDPM(m.ddpm, m.sched, 400, m.sched.T, NoClass, 1, rng, nil)
	got := MeanNNDist(x, m.data.X)
	t.Logf("DDPM: loss %.3f → %.3f; nn-dist noise %.3f, samples %.3f", first, last, base, got)
	ddim := MeanNNDist(SampleDDIM(m.ddpm, m.sched, 400, 10, NoClass, 1, rng, nil), m.data.X)
	t.Logf("DDIM 10 steps: nn-dist %.3f", ddim)
	skipQualityUnderRace(t)
	if !(last < 0.85*first) {
		t.Errorf("DDPM loss did not fall: %.3f → %.3f", first, last)
	}
	if !(got < 0.6*base) {
		t.Errorf("DDPM samples nn-dist %.3f, not < 0.6 of noise %.3f", got, base)
	}
	if !(ddim < 0.6*base) {
		t.Errorf("DDIM 10-step nn-dist %.3f, not < 0.6 of noise %.3f", ddim, base)
	}
}

func TestFlowMatchingFewSteps(t *testing.T) {
	m := toyModels()
	rng := newRNG(9)
	base := MeanNNDist(Gaussian(rng, 400), m.data.X)
	one := MeanNNDist(SampleFlow(m.flow, 400, 1, NoClass, 1, rng, nil), m.data.X)
	four := MeanNNDist(SampleFlow(m.flow, 400, 4, NoClass, 1, rng, nil), m.data.X)
	many := MeanNNDist(SampleFlow(m.flow, 400, 32, NoClass, 1, rng, nil), m.data.X)
	t.Logf("flow nn-dist: noise %.3f, 1 step %.3f, 4 steps %.3f, 32 steps %.3f", base, one, four, many)
	skipQualityUnderRace(t)
	if !(four < 0.6*base) {
		t.Errorf("4 Euler steps: nn-dist %.3f, not < 0.6 of noise %.3f", four, base)
	}
	if !(many < 0.5*base) {
		t.Errorf("32 Euler steps: nn-dist %.3f, not < half of noise %.3f", many, base)
	}
	if !(four < one) {
		t.Errorf("4 steps (%.3f) should beat 1 step (%.3f): one step lands on the mean of the data", four, one)
	}
}

// With guidance, samples of class c lie on the moon of class c, far from the other.
func TestClassifierFreeGuidance(t *testing.T) {
	m := toyModels()
	rng := newRNG(10)
	for c := range 2 {
		own, other := m.data.Subset(c), m.data.Subset(1-c)
		for _, w := range []float64{1, 3} {
			x := SampleFlow(m.cfg, 300, 16, c, w, rng, nil)
			do, dt := MeanNNDist(x, own), MeanNNDist(x, other)
			t.Logf("class %d w=%g: nn-dist own %.3f other %.3f", c, w, do, dt)
			if raceEnabled {
				continue
			}
			if !(do < 0.3*dt) {
				t.Errorf("class %d w=%g: own %.3f vs other %.3f", c, w, do, dt)
			}
		}
	}
}

// ---------- CSV ----------

func TestCSV(t *testing.T) {
	var buf bytes.Buffer
	c, err := NewCSV(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Add("data", 3, []float32{1, -2.5, 0.125, 4}, []int{0, 1}); err != nil {
		t.Fatal(err)
	}
	if err := c.Add("ddpm", 0, []float32{0.5, 0.25}, nil); err != nil {
		t.Fatal(err)
	}
	if err := c.Flush(); err != nil {
		t.Fatal(err)
	}
	want := "series,step,label,x,y\ndata,3,0,1.0000,-2.5000\ndata,3,1,0.1250,4.0000\nddpm,0,-1,0.5000,0.2500\n"
	if got := buf.String(); got != want {
		t.Fatalf("CSV:\n%s\nwant:\n%s", got, want)
	}
	if strings.Count(buf.String(), "\n") != 4 {
		t.Fatal("wrong number of rows")
	}
}

// ---------- CLIP ----------

// The 3×3 example of the lecture: cosine similarities with a clear diagonal, τ = 0.1.
func TestContrastiveLossLectureExample(t *testing.T) {
	img := autograd.NewVar(tensor.FromSlice([]float32{
		1, 0,
		0, 1,
		0.6, 0.8,
	}, 3, 2))
	txt := autograd.NewVar(tensor.FromSlice([]float32{
		0.8, 0.6,
		0, 1,
		0.6, 0.8,
	}, 3, 2))
	sim := Similarity(img, txt, 1)
	want := []float64{0.8, 0, 0.6, 0.6, 1, 0.8, 0.96, 0.8, 1}
	for i, v := range sim.Data.Data() {
		near(t, "cos", float64(v), want[i], 1e-6)
	}
	near(t, "InfoNCE τ=0.1", float64(ContrastiveLoss(img, txt, 0.1).Item()), 0.48956, 1e-4)
	near(t, "InfoNCE τ=1", float64(ContrastiveLoss(img, txt, 1).Item()), 0.92390, 1e-4)
	if RetrievalAccuracy(sim) != 1 {
		t.Errorf("retrieval accuracy %g, want 1", RetrievalAccuracy(sim))
	}
}

func TestContrastiveLossGradCheck(t *testing.T) {
	rng := newRNG(11)
	a := autograd.Param(tensor.Randn(rng, 4, 3))
	b := autograd.Param(tensor.Randn(rng, 4, 3))
	f := func() *autograd.Var { return ContrastiveLoss(a, b, 0.5) }
	if rel, where := autograd.GradCheck(f, []*autograd.Var{a, b}, 1e-2); rel > 2e-2 {
		t.Fatalf("grad check: rel err %.3g at %s", rel, where)
	}
}

func TestCLIPLearnsSharedSpace(t *testing.T) {
	rng := newRNG(12)
	pairs := MakeToyPairs(rng, 1200, 8, 24, 16, 0.1)
	c := NewCLIP(rng, 24, 16, 16, 0.1)
	test := make([]int, 200)
	for i := range test {
		test[i] = 1000 + i
	}
	train := &ToyPairs{N: 1000, DImg: 24, DTxt: 16, Img: pairs.Img[:1000*24], Txt: pairs.Txt[:1000*16]}
	img, txt := c.Encode(pairs, test)
	before := RetrievalAccuracy(Similarity(img, txt, c.Temperature))
	losses := TrainCLIP(c, train, 300, 64, 1e-2, rng)
	img, txt = c.Encode(pairs, test)
	after := RetrievalAccuracy(Similarity(img, txt, c.Temperature))
	first, last := mean(losses[:10]), mean(losses[len(losses)-20:])
	t.Logf("CLIP: loss %.3f → %.3f (ln 64 = %.3f); top-1 retrieval among 200: %.3f → %.3f",
		first, last, math.Log(64), before, after)
	if !(last < 0.3*first) {
		t.Errorf("loss did not fall enough: %.3f → %.3f", first, last)
	}
	if !(after > 0.8 && before < 0.1) {
		t.Errorf("retrieval %.3f → %.3f", before, after)
	}
}
