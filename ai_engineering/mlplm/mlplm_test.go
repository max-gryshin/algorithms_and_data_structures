package mlplm

import (
	"math"
	"math/rand/v2"
	"os"
	"testing"

	"algorithms_and_data_structures/ai_engineering/optim"
)

func loadCorpus(t *testing.T) (*Vocab, []int, []int) {
	t.Helper()
	b, err := os.ReadFile("../data/corpus_ru.txt")
	if err != nil {
		t.Fatal(err)
	}
	v := BuildVocab(string(b))
	ids := v.Encode(string(b))
	n := len(ids) * 9 / 10
	return v, ids[:n], ids[n:]
}

func TestVocabRoundTrip(t *testing.T) {
	v := BuildVocab("кот спит.")
	if got := v.Decode(v.Encode("спит кот")); got != "спит кот" {
		t.Fatalf("round trip: %q", got)
	}
}

// The MLP LM must beat the count-based bigram model of L2 on held-out text.
func TestMLPBeatsBigram(t *testing.T) {
	v, train, val := loadCorpus(t)
	bigram := BigramLoss(train, val, v.Size(), 0.01)

	rng := rand.New(rand.NewPCG(1, 1))
	cfg := Config{Context: 3, EmbDim: 10, Hidden: 128}
	m := NewModel(rng, v.Size(), cfg)
	xc, xt := Examples(train, cfg.Context)
	vc, vt := Examples(val, cfg.Context)

	// At init every character is ~equally likely: loss ≈ ln V.
	if l0 := m.Evaluate(vc, vt); math.Abs(l0-math.Log(float64(v.Size()))) > 0.1 {
		t.Fatalf("initial loss %.3f, want ≈ ln %d = %.3f", l0, v.Size(), math.Log(float64(v.Size())))
	}

	const steps = 1200
	opt := optim.NewAdamW(m.Parameters(), 3e-3, 0.01)
	sched := optim.CosineSchedule{MaxLR: 3e-3, MinLR: 3e-4, WarmupSteps: 50, TotalSteps: steps}
	for step := range steps {
		c, tg := Batch(rng, xc, xt, 64)
		opt.ZeroGrad()
		m.Loss(c, tg).Backward()
		optim.ClipGradNorm(m.Parameters(), 1)
		opt.SetLR(sched.LR(step))
		opt.Step()
	}
	if got := m.Evaluate(vc, vt); got >= bigram {
		t.Fatalf("val loss %.3f does not beat bigram %.3f", got, bigram)
	}

	out := m.Generate(rng, train[:cfg.Context], 50, 0.8)
	if len(out) != cfg.Context+50 {
		t.Fatalf("generated %d ids", len(out))
	}
}
