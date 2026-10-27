package prob

import (
	"math"
	"math/rand/v2"
	"testing"
)

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestSoftmaxSumsToOne(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for range 100 {
		z := make([]float32, 1+rng.IntN(50))
		for i := range z {
			z[i] = float32(rng.NormFloat64() * 5)
		}
		var s float64
		for _, p := range Softmax(z) {
			if p < 0 || p > 1 {
				t.Fatalf("probability out of [0,1]: %v", p)
			}
			s += float64(p)
		}
		if !near(s, 1, 1e-5) {
			t.Fatalf("sum = %v, want 1", s)
		}
	}
}

func TestSoftmaxHandExample(t *testing.T) {
	// z = [2, 1, 0]: exp = [7.389, 2.718, 1], sum 11.107 → [0.665, 0.245, 0.090].
	p := Softmax([]float64{2, 1, 0})
	want := []float64{0.6652, 0.2447, 0.0900}
	for i := range p {
		if !near(p[i], want[i], 1e-4) {
			t.Fatalf("p = %v, want %v", p, want)
		}
	}
}

func TestSoftmaxShiftInvariant(t *testing.T) {
	a := Softmax([]float64{2, 1, 0})
	b := Softmax([]float64{102, 101, 100})
	for i := range a {
		if !near(a[i], b[i], 1e-12) {
			t.Fatalf("softmax(z+c) != softmax(z): %v vs %v", a, b)
		}
	}
}

func TestSoftmaxLargeLogitsStable(t *testing.T) {
	z := []float32{1000, 999, 998}
	naive := SoftmaxNaive(z)
	if !math.IsNaN(float64(naive[0])) {
		t.Fatalf("naive softmax should overflow to NaN, got %v", naive)
	}
	p := Softmax(z)
	want := Softmax([]float32{2, 1, 0})
	for i := range p {
		if math.IsNaN(float64(p[i])) || !near(float64(p[i]), float64(want[i]), 1e-6) {
			t.Fatalf("stable softmax = %v, want %v", p, want)
		}
	}
	if lse := LogSumExp(z); !near(lse, 1000+math.Log(1+math.Exp(-1)+math.Exp(-2)), 1e-9) {
		t.Fatalf("LogSumExp = %v", lse)
	}
	ls := LogSoftmax([]float64{0, -1000})
	if !near(ls[1], -1000, 1e-9) {
		t.Fatalf("LogSoftmax must stay finite where softmax underflows: %v", ls)
	}
}

func TestSoftmaxTemperature(t *testing.T) {
	z := []float64{2, 1, 0}
	if p := SoftmaxTemperature(z, 0); p[0] != 1 || p[1] != 0 {
		t.Fatalf("T=0 must be greedy: %v", p)
	}
	cold, hot := SoftmaxTemperature(z, 0.5), SoftmaxTemperature(z, 2)
	if !(cold[0] > Softmax(z)[0] && Softmax(z)[0] > hot[0]) {
		t.Fatalf("temperature must sharpen/flatten: %v %v", cold, hot)
	}
	if !(Entropy(cold) < Entropy(z2p(z)) && Entropy(z2p(z)) < Entropy(hot)) {
		t.Fatal("entropy must grow with temperature")
	}
}

func z2p(z []float64) []float64 { return Softmax(z) }

func TestEntropyCoinAndDice(t *testing.T) {
	cases := []struct {
		p    []float64
		bits float64
	}{
		{[]float64{0.5, 0.5}, 1},
		{[]float64{0.9, 0.1}, 0.4690},
		{[]float64{1, 0}, 0},
		{[]float64{1. / 6, 1. / 6, 1. / 6, 1. / 6, 1. / 6, 1. / 6}, math.Log2(6)},
	}
	for _, c := range cases {
		if got := NatsToBits(Entropy(c.p)); !near(got, c.bits, 1e-4) {
			t.Fatalf("H(%v) = %v bits, want %v", c.p, got, c.bits)
		}
	}
}

func TestCrossEntropyAndKL(t *testing.T) {
	p := []float64{0.5, 0.5}
	q := []float64{0.9, 0.1}
	// H(p,q) = −0.5 ln 0.9 − 0.5 ln 0.1 = 1.2040 nats; H(p) = ln 2 = 0.6931; KL = 0.5108.
	if h := CrossEntropy(p, q); !near(h, 1.2040, 1e-4) {
		t.Fatalf("H(p,q) = %v", h)
	}
	if d := KL(p, q); !near(d, CrossEntropy(p, q)-Entropy(p), 1e-12) || !near(d, 0.5108, 1e-4) {
		t.Fatalf("KL = %v", d)
	}
	if d := KL(q, p); !near(d, 0.3681, 1e-4) {
		t.Fatalf("KL(q,p) = %v (KL is not symmetric)", d)
	}
	if KL(p, p) != 0 {
		t.Fatal("KL(p,p) must be 0")
	}
}

func TestCrossEntropyOneHot(t *testing.T) {
	q := []float32{0.7, 0.2, 0.1}
	onehot := []float32{0, 1, 0}
	if h := CrossEntropy(onehot, q); !near(h, -math.Log(0.2), 1e-6) {
		t.Fatalf("CE(onehot, q) = %v, want −ln 0.2 = %v", h, -math.Log(0.2))
	}
	z := []float32{3, 1, -2}
	want := -math.Log(float64(Softmax(z)[1]))
	if got := CrossEntropyLogits(z, 1); !near(got, want, 1e-6) {
		t.Fatalf("CrossEntropyLogits = %v, want %v", got, want)
	}
	if got := CrossEntropyLogits([]float32{1000, 0}, 1); !near(got, 1000, 1e-3) {
		t.Fatalf("CrossEntropyLogits must not overflow: %v", got)
	}
}

func TestPerplexity(t *testing.T) {
	// A uniform guess among 4 options: NLL = ln 4 → perplexity 4.
	if p := Perplexity(math.Log(4)); !near(p, 4, 1e-12) {
		t.Fatalf("perplexity = %v", p)
	}
	// 8 bytes with total NLL = 8·ln 2 nats = 8 bits → 1 bit per byte.
	if b := BitsPerByte(8*math.Ln2, 8); !near(b, 1, 1e-12) {
		t.Fatalf("bpb = %v", b)
	}
}

func TestSampleFrequencies(t *testing.T) {
	probs := []float32{0.5, 0.3, 0.15, 0.05, 0}
	const n = 200_000
	rng := rand.New(rand.NewPCG(42, 7))
	cdf := NewCDF(probs)
	freq := make([]int, len(probs))
	freqCDF := make([]int, len(probs))
	for range n {
		freq[Sample(rng, probs)]++
		freqCDF[cdf.Sample(rng)]++
	}
	for i, p := range probs {
		// Standard error sqrt(p(1−p)/n) ≤ 0.0012; allow ~4σ.
		for _, f := range [][]int{freq, freqCDF} {
			if got := float64(f[i]) / n; !near(got, float64(p), 0.005) {
				t.Fatalf("freq[%d] = %v, want ≈ %v", i, got, p)
			}
		}
	}
	if freq[4] != 0 || freqCDF[4] != 0 {
		t.Fatal("zero-probability outcome was sampled")
	}
}
