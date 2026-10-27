package generate

import (
	"context"
	"encoding/json"
	"math"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
	"unicode/utf8"

	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/tokenizer"
)

// The worked example of L10: logits of five candidate words after «Кот сидел на».
var lectureLogits = []float32{2.0, 1.5, 1.0, 0.0, -1.0} // крыльце, окне, солнце, столе, луне

func near(t *testing.T, name string, got []float64, want ...float64) {
	t.Helper()
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-4 {
			t.Fatalf("%s = %.4f, want %v", name, got, want)
		}
	}
}

func TestSamplerLectureExample(t *testing.T) {
	dist := func(c SamplerConfig, hist ...int) []float64 { return NewSampler(c).Distribution(lectureLogits, hist) }
	near(t, "greedy", dist(SamplerConfig{}), 1, 0, 0, 0, 0)
	near(t, "T=1", dist(SamplerConfig{Temperature: 1}), 0.4631, 0.2809, 0.1704, 0.0627, 0.0231)
	near(t, "T=0.5", dist(SamplerConfig{Temperature: 0.5}), 0.6562, 0.2414, 0.0888, 0.0120, 0.0016)
	near(t, "T=2", dist(SamplerConfig{Temperature: 2}), 0.3360, 0.2617, 0.2038, 0.1236, 0.0750)
	near(t, "top-k 2", dist(SamplerConfig{Temperature: 1, TopK: 2}), 0.6225, 0.3775, 0, 0, 0)
	near(t, "top-p 0.8", dist(SamplerConfig{Temperature: 1, TopP: 0.8}), 0.5065, 0.3072, 0.1863, 0, 0)
	near(t, "min-p 0.1", dist(SamplerConfig{Temperature: 1, MinP: 0.1}), 0.4740, 0.2875, 0.1744, 0.0641, 0)
	// «крыльце» (0) already said twice, «солнце» (2) once.
	z := []float64{2, 1.5, 1, 0, -1}
	ApplyPenalties(z, []int{0, 2, 0}, 1.3, 0, 0)
	near(t, "repetition 1.3", z, 1.5385, 1.5, 0.7692, 0, -1)
	z = []float64{2, 1.5, 1, 0, -1}
	ApplyPenalties(z, []int{0, 2, 0}, 0, 0.5, 0.5)
	near(t, "frequency+presence 0.5", z, 0.5, 1.5, 0, 0, -1)
	// A negative logit is pushed further down by the repetition penalty: −1·1.3.
	z = []float64{-1, 0}
	ApplyPenalties(z, []int{0}, 1.3, 0, 0)
	near(t, "repetition of a negative logit", z, -1.3, 0)
}

func TestDrawMatchesDistribution(t *testing.T) {
	p := NewSampler(SamplerConfig{Temperature: 1}).Distribution(lectureLogits, nil)
	rng := rand.New(rand.NewPCG(1, 2))
	counts := make([]float64, len(p))
	n := 200_000
	for range n {
		counts[Draw(rng, p)]++
	}
	for i := range p {
		if math.Abs(counts[i]/float64(n)-p[i]) > 0.005 {
			t.Fatalf("token %d drawn %.4f of the time, p = %.4f", i, counts[i]/float64(n), p[i])
		}
	}
}

// For d ~ q, the token returned by Verify is distributed as p — the theorem that
// makes speculative decoding exact.
func TestVerifyPreservesTargetDistribution(t *testing.T) {
	p := []float64{0.5, 0.3, 0.2, 0}
	q := []float64{0.2, 0.2, 0.3, 0.3}
	rng := rand.New(rand.NewPCG(3, 4))
	counts := make([]float64, len(p))
	accepted := 0
	n := 200_000
	for range n {
		d := Draw(rng, q)
		tok, ok := Verify(rng, p, q, d)
		counts[tok]++
		if ok {
			accepted++
		}
	}
	for i := range p {
		if math.Abs(counts[i]/float64(n)-p[i]) > 0.005 {
			t.Fatalf("token %d: %.4f, want %.4f", i, counts[i]/float64(n), p[i])
		}
	}
	// Acceptance rate α = Σ min(p, q) = 0.2 + 0.2 + 0.2 + 0 = 0.6.
	if a := float64(accepted) / float64(n); math.Abs(a-0.6) > 0.005 {
		t.Fatalf("acceptance %.4f, want 0.6", a)
	}
	if e := ExpectedTokensPerRound(0.8, 4); math.Abs(e-3.3616) > 1e-4 {
		t.Fatalf("E[tokens] = %v", e)
	}
}

func TestDetokenizerHoldsIncompleteCharacters(t *testing.T) {
	text := "ёж и 🦔, ok"
	b := []byte(text)
	for split := 1; split <= 3; split++ { // feed the bytes in pieces of 1, 2, 3
		var d Detokenizer
		var out strings.Builder
		for i := 0; i < len(b); i += split {
			s := d.Push(b[i:min(i+split, len(b))])
			if !utf8.ValidString(s) {
				t.Fatalf("emitted invalid UTF-8 %q", s)
			}
			out.WriteString(s)
		}
		out.WriteString(d.Flush())
		if out.String() != text {
			t.Fatalf("got %q", out.String())
		}
	}
	var d Detokenizer
	if s := d.Push([]byte{0xD1}); s != "" { // first byte of «ё»
		t.Fatalf("emitted %q before the character was complete", s)
	}
	if s := d.Push([]byte{0x91, 'x'}); s != "ёx" {
		t.Fatalf("got %q", s)
	}
}

func TestStopSequenceAcrossTokens(t *testing.T) {
	s := stopper{stops: []string{"\n\nUser:", "END"}}
	var out strings.Builder
	for _, piece := range []string{"Привет", "!\n", "\nUs", "er: что"} {
		emit, stopped := s.push(piece)
		out.WriteString(emit)
		if stopped {
			if out.String() != "Привет!" {
				t.Fatalf("emitted %q", out.String())
			}
			return
		}
		if strings.Contains(out.String(), "\n\nU") {
			t.Fatalf("a possible stop prefix was emitted: %q", out.String())
		}
	}
	t.Fatal("stop sequence not detected")
}

// ---------- generation with a model ----------

// byteTok is a 256-token byte-level tokenizer: any string is representable.
type byteTok struct{}

func (byteTok) Encode(s string) []int {
	ids := make([]int, len(s))
	for i := range len(s) {
		ids[i] = int(s[i])
	}
	return ids
}
func (byteTok) TokenBytes(id int) []byte { return []byte{byte(id)} }
func (byteTok) VocabSize() int           { return 256 }

func tinyModel(seed uint64, vocab int, layers int) *gpt.Model {
	cfg := gpt.TinyLlama(vocab, 64)
	cfg.Layers = layers
	return gpt.New(rand.New(rand.NewPCG(seed, seed)), cfg)
}

// The KV-cache must not change the output: greedy and sampled generations with and
// without the cache are identical token for token — also past the context (64), where
// the window slides.
func TestCachedGenerationEqualsUncached(t *testing.T) {
	m := tinyModel(1, 256, 2)
	for _, sc := range []SamplerConfig{{}, {Temperature: 1, TopK: 20, Seed: 7}, {Temperature: 0.8, TopP: 0.9, RepetitionPenalty: 1.2, Seed: 3}} {
		cfg := Config{MaxTokens: 100, Sampler: sc}
		a, _, err := IDs(context.Background(), m, byteTok{}, []int{'h', 'i'}, cfg)
		if err != nil {
			t.Fatal(err)
		}
		cfg.NoCache = true
		b, _, _ := IDs(context.Background(), m, byteTok{}, []int{'h', 'i'}, cfg)
		if !slices.Equal(a, b) || len(a) != 100 {
			t.Fatalf("%+v: cached %v\nuncached %v", sc, a, b)
		}
	}
}

func TestGenerateStreamsStopsAndCancels(t *testing.T) {
	m := tinyModel(2, 256, 1)
	var text strings.Builder
	n := 0
	for tok, err := range Generate(context.Background(), m, byteTok{}, "ab", Config{MaxTokens: 30, Sampler: SamplerConfig{Temperature: 1, Seed: 1}}) {
		if err != nil {
			t.Fatal(err)
		}
		text.WriteString(tok.Text)
		n++
		if n == 5 {
			break // the consumer may stop early
		}
	}
	if n != 5 {
		t.Fatalf("got %d tokens", n)
	}
	// Stop sequence: force one by generating greedily and stopping on the text it produces.
	full, _, _ := Text(context.Background(), m, byteTok{}, "ab", Config{MaxTokens: 20})
	stop := full[3:5] // matching is on bytes, like the detokenized stream
	got, st, _ := Text(context.Background(), m, byteTok{}, "ab", Config{MaxTokens: 20, Stop: []string{stop}})
	if st.Finish != "stop" || got != full[:strings.Index(full, stop)] {
		t.Fatalf("stop %q: got %q (%s) from %q", stop, got, st.Finish, full)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := Text(ctx, m, byteTok{}, "ab", Config{MaxTokens: 5}); err == nil {
		t.Fatal("cancelled context ignored")
	}
}

// Constrained decoding: an untrained model, whatever it samples, outputs valid JSON.
func TestJSONConstraint(t *testing.T) {
	m := tinyModel(3, 256, 1)
	done := 0
	for seed := range uint64(8) {
		c := &JSON{MaxDepth: 3, MaxString: 8}
		out, st, err := Text(context.Background(), m, byteTok{}, "{", Config{MaxTokens: 300, Constraint: c,
			Sampler: SamplerConfig{Temperature: 1, Seed: seed}})
		if err != nil {
			t.Fatal(err)
		}
		// The prompt is not part of the constrained output; we constrained from scratch.
		if st.Finish == "constraint" {
			done++
			if !json.Valid([]byte(out)) {
				t.Fatalf("seed %d: invalid JSON %q", seed, out)
			}
		}
	}
	if done == 0 {
		t.Fatal("no generation finished within 300 tokens")
	}
}

func TestSchemaConstraint(t *testing.T) {
	m := tinyModel(4, 256, 1)
	// A BPE tokenizer: multi-byte tokens must be checked byte by byte against the automaton.
	tok, err := tokenizer.Train(`{"name":"Анна","age":42,"admin":true} {"name":"Олег","age":7,"admin":false}`, 300,
		tokenizer.Options{Pattern: tokenizer.PatternGPT4})
	if err != nil {
		t.Fatal(err)
	}
	m2 := tinyModel(5, tok.VocabSize(), 1)
	for _, c := range []struct {
		m   *gpt.Model
		tok Tokenizer
	}{{m, byteTok{}}, {m2, tok}} {
		for seed := range uint64(10) {
			s := &Schema{Fields: []Field{{"name", String}, {"age", Int}, {"admin", Bool}}, MaxString: 12}
			out, st, err := Text(context.Background(), c.m, c.tok, "x", Config{MaxTokens: 200, Constraint: s,
				Sampler: SamplerConfig{Temperature: 1, Seed: seed}})
			if err != nil {
				t.Fatal(err)
			}
			var v struct {
				Name  string `json:"name"`
				Age   int    `json:"age"`
				Admin bool   `json:"admin"`
			}
			if st.Finish != "constraint" || json.Unmarshal([]byte(out), &v) != nil || !utf8.ValidString(out) {
				t.Fatalf("seed %d: %q (%s)", seed, out, st.Finish)
			}
		}
	}
}

// Greedy speculative decoding returns exactly the target's greedy output.
func TestSpeculativeGreedyEqualsTarget(t *testing.T) {
	target, draft := tinyModel(6, 256, 3), tinyModel(7, 256, 1)
	prompt := []int{'k', 'o', 't'}
	want, _, err := IDs(context.Background(), target, byteTok{}, prompt, Config{MaxTokens: 40})
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []int{1, 3, 5} {
		got, st, err := Speculative(context.Background(), target, draft, prompt, k, 40, SamplerConfig{})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got, want) {
			t.Fatalf("k=%d: speculative %v\ntarget %v", k, got, want)
		}
		if st.Tokens < 40 || st.Rounds == 0 {
			t.Fatalf("k=%d: stats %+v", k, st)
		}
	}
	// A draft identical to the target is always accepted: k+1 tokens per round.
	_, st, _ := Speculative(context.Background(), target, target, prompt, 4, 40, SamplerConfig{Temperature: 1, Seed: 1})
	if st.AcceptRate != 1 || st.TokensRound != 5 {
		t.Fatalf("self-draft: %+v", st)
	}
}
