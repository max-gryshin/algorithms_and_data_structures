package gpt

import (
	"math"
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/nn"
	"algorithms_and_data_structures/ai_engineering/optim"
	"algorithms_and_data_structures/ai_engineering/scaling"
)

func randIDs(rng *rand.Rand, b, t, v int) [][]int {
	ids := make([][]int, b)
	for i := range ids {
		ids[i] = make([]int, t)
		for j := range ids[i] {
			ids[i][j] = rng.IntN(v)
		}
	}
	return ids
}

func allConfigs(v, ctx int) []Config {
	sin := TinyGPT2(v, ctx)
	sin.Name, sin.Pos = "tiny-sinusoidal", PosSinusoidal
	nope := TinyLlama(v, ctx)
	nope.Name, nope.Pos = "tiny-nope", PosNone
	return []Config{TinyGPT2(v, ctx), TinyLlama(v, ctx), sin, nope}
}

func TestForwardShape(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	for _, cfg := range allConfigs(50, 16) {
		m := New(rng, cfg)
		logits := m.Forward(randIDs(rng, 3, 7, 50))
		if s := logits.Shape(); len(s) != 3 || s[0] != 3 || s[1] != 7 || s[2] != 50 {
			t.Fatalf("%s: logits shape %v, want [3 7 50]", cfg.Name, s)
		}
	}
}

// The built model and the formula must agree for every architecture variant.
func TestParamCountMatchesFormula(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2))
	for _, cfg := range allConfigs(50, 16) {
		for _, tied := range []bool{true, false} {
			cfg.Tied = tied
			if got, want := New(rng, cfg).NumParams(), cfg.Params().Total; int64(got) != want {
				t.Errorf("%s tied=%v: model has %d params, formula says %d", cfg.Name, tied, got, want)
			}
		}
	}
}

// Exact counts of real models (Hugging Face) and agreement with the L11 calculator.
func TestRealModelParams(t *testing.T) {
	for _, tc := range []struct {
		cfg  Config
		sc   scaling.ModelConfig
		want int64
	}{
		{GPT2Small, scaling.GPT2Small, 124_439_808},
		{Llama2_7B, scaling.Llama2_7B, 6_738_415_616},
		{Llama3_8B, scaling.Llama3_8B, 8_030_261_248},
	} {
		if got := tc.cfg.Params().Total; got != tc.want {
			t.Errorf("%s: %d params, want %d", tc.cfg.Name, got, tc.want)
		}
		if got := tc.sc.Params(); got != tc.want {
			t.Errorf("%s: scaling package says %d", tc.cfg.Name, got)
		}
	}
	// 12·L·d² + V·d for GPT-2 small: 84 934 656 + 38 597 376.
	if got := GPT2Small.ApproxParams(); got != 123_532_032 {
		t.Errorf("GPT-2 approx = %d", got)
	}
	if got := SwiGLUHidden(4096, 256); got != 11008 {
		t.Errorf("SwiGLUHidden(4096) = %d, want 11008", got)
	}
	// KV-cache per token: Llama-2-7B (MHA) 512 KiB, Llama-3-8B (GQA 32/8) 128 KiB.
	if a, b := Llama2_7B.Memory().KVPerToken, Llama3_8B.Memory().KVPerToken; a != 512<<10 || b != 128<<10 {
		t.Errorf("KV per token = %v, %v", a, b)
	}
}

// Before training the prediction is almost uniform, so loss ≈ −ln(1/V) = ln V (L8).
func TestInitialLossIsLnV(t *testing.T) {
	const v = 64
	rng := rand.New(rand.NewPCG(3, 3))
	for _, cfg := range allConfigs(v, 16) {
		m := New(rng, cfg)
		loss := float64(m.Loss(randIDs(rng, 4, 16, v), randIDs(rng, 4, 16, v)).Item())
		if math.Abs(loss-math.Log(v)) > 0.05 {
			t.Errorf("%s: initial loss %.4f, want ln %d = %.4f", cfg.Name, loss, v, math.Log(v))
		}
	}
}

// The classic sanity check: a model must be able to memorize a single batch.
// If the loss does not go to ~0, something in forward/backward/optimizer is broken.
func TestOverfitSingleBatch(t *testing.T) {
	const v, ctx = 32, 12
	for _, cfg := range []Config{TinyGPT2(v, ctx), TinyLlama(v, ctx)} {
		rng := rand.New(rand.NewPCG(4, 4))
		m := New(rng, cfg)
		seq := randIDs(rng, 2, ctx+1, v)
		ids, targets := make([][]int, 2), make([][]int, 2)
		for i, s := range seq {
			ids[i], targets[i] = s[:ctx], s[1:] // targets = inputs shifted by one
		}
		ps := m.Parameters()
		opt := optim.NewAdamW(ps, 1e-2, 0)
		var first, last float32
		for step := range 150 {
			opt.ZeroGrad()
			loss := m.Loss(ids, targets)
			loss.Backward()
			optim.ClipGradNorm(ps, 1)
			opt.Step()
			if step == 0 {
				first = loss.Item()
			}
			last = loss.Item()
		}
		t.Logf("%s: loss %.3f → %.4f", cfg.Name, first, last)
		if last > 0.05 {
			t.Errorf("%s: failed to overfit one batch: loss %.3f → %.4f", cfg.Name, first, last)
		}
	}
}

// Tied head: the token embedding receives gradient from both the input lookup and the LM head.
func TestTiedHeadSharesWeights(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 5))
	m := New(rng, TinyGPT2(20, 8))
	if m.Head != nil {
		t.Fatal("tied model must not have a separate head")
	}
	ids := [][]int{{1, 2, 3}}
	m.Loss(ids, [][]int{{2, 3, 4}}).Backward()
	g := m.Tok.W.Grad.Data()
	// Token 19 never appears in the input, yet its row gets gradient through the head (softmax over all V).
	row := g[19*32 : 20*32]
	nonzero := false
	for _, x := range row {
		nonzero = nonzero || x != 0
	}
	if !nonzero {
		t.Fatal("unused token row got no gradient: head is not tied")
	}
	if m.Cfg.Norm != nn.NormLayer {
		t.Fatal("TinyGPT2 must use LayerNorm")
	}
}

func TestValidate(t *testing.T) {
	bad := TinyLlama(10, 8)
	bad.KVHeads = 3
	if bad.Validate() == nil {
		t.Fatal("h=4, h_kv=3 must be rejected")
	}
	for _, c := range []Config{GPT2Small, Llama2_7B, Llama3_8B} {
		if err := c.Validate(); err != nil {
			t.Fatal(err)
		}
	}
}

// End-to-end gradient check of a whole (very small) model of each family.
func TestModelGradCheck(t *testing.T) {
	for _, cfg := range allConfigs(10, 6) {
		kv := 2 // MHA stays MHA, GQA becomes MQA (one K/V head for both query heads)
		if cfg.KVHeads < cfg.Heads {
			kv = 1
		}
		cfg.DModel, cfg.Heads, cfg.KVHeads, cfg.Layers, cfg.FFHidden = 8, 2, kv, 1, 12
		cfg.InitStd = 0.3 // larger weights: gradients well above float32 noise
		rng := rand.New(rand.NewPCG(6, 6))
		m := New(rng, cfg)
		ids, targets := randIDs(rng, 2, 4, 10), randIDs(rng, 2, 4, 10)
		var ps []*autograd.Var
		for _, p := range m.Parameters() {
			if !isKeyBias(m, p) {
				ps = append(ps, p)
			}
		}
		rel, where := autograd.GradCheck(func() *autograd.Var { return m.Loss(ids, targets) }, ps, 5e-3)
		if rel > 3e-2 {
			t.Errorf("%s: gradcheck %.3g at %s", cfg.Name, rel, where)
		}
	}
}

// isKeyBias reports whether p is a W_K bias: its true gradient is 0 (softmax is shift-invariant).
func isKeyBias(m *Model, p *autograd.Var) bool {
	for _, b := range m.Blocks {
		if p == b.Attn.K.B {
			return true
		}
	}
	return false
}
