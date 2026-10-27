package scaling

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"
	"testing"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

func TestParamsKnownModels(t *testing.T) {
	// GPT-2 small: exactly 124,439,808 parameters (HF transformers count).
	if got := GPT2Small.Params(); got != 124_439_808 {
		t.Fatalf("GPT-2 small params = %d, want 124439808", got)
	}
	// Llama 2 7B: 6,738,415,616 parameters (HF count).
	if got := Llama2_7B.Params(); got != 6_738_415_616 {
		t.Fatalf("Llama-2-7B params = %d, want 6738415616", got)
	}
	// Llama 3 8B: 8,030,261,248 parameters (HF count).
	if got := Llama3_8B.Params(); got != 8_030_261_248 {
		t.Fatalf("Llama-3-8B params = %d, want 8030261248", got)
	}
}

func TestKVCache(t *testing.T) {
	// Llama-2-7B in fp16: 2·32·32·128·2 = 524,288 bytes = 0.5 MiB per token.
	if got := Llama2_7B.KVCacheBytesPerToken(2); got != 524_288 {
		t.Fatalf("KV bytes/token = %d", got)
	}
	// GQA with 8 KV heads makes Llama-3-8B four times cheaper per token.
	if got := Llama3_8B.KVCacheBytesPerToken(2); got != 131_072 {
		t.Fatalf("KV bytes/token = %d", got)
	}
}

func TestTrainingMemory(t *testing.T) {
	m := MixedPrecisionAdam(1)
	if m.Total() != 16 {
		t.Fatalf("bytes/param = %v, want 16", m.Total())
	}
	const n = 7e9
	if got := ZeROBytesPerGPU(n, 64, 0); got != 112e9 {
		t.Fatalf("stage 0 = %v", got)
	}
	if got := ZeROBytesPerGPU(n, 64, 3); got != 112e9/64 {
		t.Fatalf("stage 3 = %v", got)
	}
	for s := 1; s <= 3; s++ {
		if ZeROBytesPerGPU(n, 64, s) >= ZeROBytesPerGPU(n, 64, s-1) {
			t.Fatalf("stage %d must use less memory than stage %d", s, s-1)
		}
	}
}

func TestChinchilla(t *testing.T) {
	// Chinchilla itself: 70B params × 1.4T tokens ≈ 5.88e23 FLOPs.
	if c := TrainingFLOPs(70e9, 1.4e12); math.Abs(c-5.88e23) > 1e21 {
		t.Fatalf("C = %g", c)
	}
	n, d := ComputeOptimal20(1e21)
	if math.Abs(n-2.887e9)/2.887e9 > 1e-3 || math.Abs(d/n-20) > 1e-9 {
		t.Fatalf("1e21: N=%g D=%g", n, d)
	}
	// The analytic optimum of the fit must actually minimize the loss on the budget curve.
	const c = 1e23
	nOpt, dOpt := ChinchillaFit.Optimal(c)
	best := ChinchillaFit.Loss(nOpt, dOpt)
	for _, f := range []float64{0.5, 0.8, 1.25, 2} {
		n := nOpt * f
		if ChinchillaFit.Loss(n, c/6/n) < best {
			t.Fatalf("N=%g beats the analytic optimum %g", n, nOpt)
		}
	}
	// More data at fixed N must lower the loss, but never below E.
	if !(ChinchillaFit.Loss(1e9, 1e12) < ChinchillaFit.Loss(1e9, 1e11)) || ChinchillaFit.Loss(1e15, 1e15) < ChinchillaFit.E {
		t.Fatal("loss is not monotone or drops below E")
	}
}

func TestGPUHours(t *testing.T) {
	// 1e21 FLOPs on a 1e15 FLOP/s GPU at 40% MFU: 1e21/4e14 = 2.5e6 s ≈ 694 h.
	if h := GPUHours(1e21, 1e15, 0.4); math.Abs(h-694.44) > 0.01 {
		t.Fatalf("hours = %v", h)
	}
}

func TestJaccardByHand(t *testing.T) {
	a := Shingles("кот сидит на ковре", 2) // {кот сидит, сидит на, на ковре}
	b := Shingles("кот сидит на диване", 2)
	// ∩ = {кот сидит, сидит на}, ∪ = 4 shingles → 2/4.
	if got := Jaccard(a, b); got != 0.5 {
		t.Fatalf("Jaccard = %v, want 0.5", got)
	}
}

func TestMinHashEstimate(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	words := make([]string, 400)
	for i := range words {
		words[i] = fmt.Sprintf("w%d", rng.IntN(100000))
	}
	base := strings.Join(words, " ")
	words[100], words[250] = "zzz", "yyy" // a small edit
	edited := strings.Join(words, " ")

	a, b := Shingles(base, 3), Shingles(edited, 3)
	exact := Jaccard(a, b)
	mh := NewMinHasher(512, 7)
	est := EstimateJaccard(mh.Signature(a), mh.Signature(b))
	// Standard error is √(J(1−J)/k) ≈ 0.01 here; allow 4σ.
	if math.Abs(est-exact) > 0.05 {
		t.Fatalf("estimate %.3f vs exact %.3f", est, exact)
	}
}

func TestLSHProbability(t *testing.T) {
	// b=20, r=5: s=0.8 → 1 − (1 − 0.32768)^20 ≈ 0.99964; s=0.3 → 1 − (1 − 0.00243)^20 ≈ 0.0475.
	if p := LSHProbability(0.8, 20, 5); math.Abs(p-0.99964) > 1e-4 {
		t.Fatalf("p(0.8) = %v", p)
	}
	if p := LSHProbability(0.3, 20, 5); math.Abs(p-0.0475) > 1e-3 {
		t.Fatalf("p(0.3) = %v", p)
	}
}

func TestNearDuplicates(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	doc := func() []string {
		w := make([]string, 200)
		for i := range w {
			w[i] = fmt.Sprintf("t%d", rng.IntN(50000))
		}
		return w
	}
	a, c := doc(), doc()
	b := append([]string(nil), a...)
	b[10], b[150] = "edit1", "edit2"
	docs := []string{strings.Join(a, " "), strings.Join(b, " "), strings.Join(c, " ")}

	got := NearDuplicates(docs, 3, 20, 5, 0.8, 42)
	if len(got) != 1 || got[0] != [2]int{0, 1} {
		t.Fatalf("near duplicates = %v, want [[0 1]]", got)
	}
}

func TestRingAllReduce(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, n := range []int{1, 2, 4, 7} {
		const size = 1003 // not divisible by n on purpose
		inputs := make([][]float32, n)
		want := make([]float32, size)
		for i := range inputs {
			inputs[i] = make([]float32, size)
			for j := range inputs[i] {
				v := float32(rng.IntN(100)) // integers: exact in float32, any summation order
				inputs[i][j] = v
				want[j] += v
			}
		}
		out, bytes := RingAllReduce(inputs)
		for i := range out {
			for j := range want {
				if out[i][j] != want[j] {
					t.Fatalf("n=%d worker %d elem %d: %v != %v", n, i, j, out[i][j], want[j])
				}
			}
		}
		// Worker 0 sends 2·(n−1) chunks; chunk sizes differ by at most 1 element.
		ideal := 2 * float64(n-1) / float64(n) * size * 4
		if math.Abs(float64(bytes)-ideal) > float64(2*(n-1)*4) {
			t.Fatalf("n=%d: worker 0 sent %d bytes, ideal %.0f", n, bytes, ideal)
		}
	}
}

func TestMoERouting(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 8))
	const d, e, tok = 8, 4, 64
	m := &MoE{Router: tensor.Randn(rng, d, e), K: 2}
	for range e {
		m.Experts = append(m.Experts, tensor.Randn(rng, d, d))
	}
	x := tensor.Randn(rng, tok, d)
	y, r := m.Forward(x)
	if s := y.Shape(); s[0] != tok || s[1] != d {
		t.Fatalf("shape %v", s)
	}
	var load, prob float64
	for i := range tok {
		var w float32
		for _, v := range r.Weights[i] {
			w += v
		}
		if math.Abs(float64(w)-1) > 1e-5 || r.Experts[i][0] == r.Experts[i][1] {
			t.Fatalf("token %d: weights %v experts %v", i, r.Weights[i], r.Experts[i])
		}
	}
	for j := range e {
		load += r.Load[j]
		prob += r.Prob[j]
	}
	if math.Abs(load-1) > 1e-9 || math.Abs(prob-1) > 1e-9 {
		t.Fatalf("Σf=%v ΣP=%v", load, prob)
	}

	// A zero router gives uniform probabilities, so the aux loss is exactly 1.
	m.Router = tensor.New(d, e)
	if r := m.Route(x); math.Abs(r.AuxLoss-1) > 1e-9 {
		t.Fatalf("uniform aux loss = %v", r.AuxLoss)
	}
}
