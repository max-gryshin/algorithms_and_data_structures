// Command generate samples text from a GoGPT checkpoint (lecture L10).
//
//	go run ./ai_engineering/gpt/cmd/generate -ckpt ai_engineering/data/runs/corpus_ru-char/model.ckpt -prompt "Кот" -temp 0.8
//	go run ./ai_engineering/gpt/cmd/generate -ckpt … -temp 0               # greedy
//	go run ./ai_engineering/gpt/cmd/generate -ckpt … -json schema -prompt "x"
//	go run ./ai_engineering/gpt/cmd/generate -ckpt big.ckpt -draft small.ckpt -k 4 -temp 0
//	go run ./ai_engineering/gpt/cmd/generate -ckpt … -bench                # KV-cache vs recomputation
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"time"

	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/gpt/generate"
	"algorithms_and_data_structures/ai_engineering/gpt/train"
)

type stops []string

func (s *stops) String() string     { return strings.Join(*s, ",") }
func (s *stops) Set(v string) error { *s = append(*s, strings.ReplaceAll(v, `\n`, "\n")); return nil }

func main() {
	ckpt := flag.String("ckpt", "ai_engineering/data/runs/corpus_ru-char/model.ckpt", "checkpoint written by cmd/train")
	prompt := flag.String("prompt", "Кот", "prompt")
	n := flag.Int("n", 200, "tokens to generate")
	temp := flag.Float64("temp", 0.8, "temperature (0 = greedy)")
	topK := flag.Int("topk", 0, "top-k (0 = off)")
	topP := flag.Float64("topp", 0, "top-p / nucleus (0 = off)")
	minP := flag.Float64("minp", 0, "min-p (0 = off)")
	rep := flag.Float64("rep", 0, "repetition penalty (CTRL), e.g. 1.3")
	freq := flag.Float64("freq", 0, "frequency penalty")
	pres := flag.Float64("pres", 0, "presence penalty")
	seed := flag.Uint64("seed", 1, "sampling seed")
	var stop stops
	flag.Var(&stop, "stop", `stop sequence (repeatable; "\n" allowed)`)
	jsonMode := flag.String("json", "", `constrained decoding: "any" (any JSON object/array) or "schema" ({"name":str,"age":int})`)
	draftPath := flag.String("draft", "", "draft checkpoint for speculative decoding")
	k := flag.Int("k", 4, "draft tokens per speculative round")
	noCache := flag.Bool("nocache", false, "recompute the whole sequence every step")
	bench := flag.Bool("bench", false, "measure prefill and decode speed with and without the KV-cache")
	flag.Parse()

	c, err := gpt.LoadCheckpointFile(*ckpt)
	if err != nil {
		log.Fatal(err)
	}
	tok, err := train.TokenizerFromMeta(c.Meta)
	if err != nil {
		log.Fatal(err)
	}
	m := c.Model
	fmt.Fprintf(os.Stderr, "%s: %d params, V=%d, context %d, trained %s steps\n", m.Cfg.Name, m.NumParams(),
		m.Cfg.Vocab, m.Cfg.Context, c.Meta["step"])
	if *bench {
		benchmark(m)
		return
	}
	sc := generate.SamplerConfig{Temperature: *temp, TopK: *topK, TopP: *topP, MinP: *minP,
		RepetitionPenalty: *rep, FrequencyPenalty: *freq, PresencePenalty: *pres, Seed: *seed}
	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	if *draftPath != "" {
		speculative(ctx, m, *draftPath, tok, *prompt, *n, *k, sc)
		return
	}
	cfg := generate.Config{MaxTokens: *n, Sampler: sc, Stop: stop, NoCache: *noCache}
	switch *jsonMode {
	case "":
	case "any":
		cfg.Constraint = &generate.JSON{MaxDepth: 3, MaxString: 24}
	case "schema":
		cfg.Constraint = &generate.Schema{Fields: []generate.Field{{Name: "name", Kind: generate.String},
			{Name: "age", Kind: generate.Int}}, MaxString: 24}
	default:
		log.Fatalf("unknown -json %q", *jsonMode)
	}
	if cfg.Constraint != nil {
		*prompt = strings.TrimSpace(*prompt) // the constrained output starts right after the prompt
	} else {
		fmt.Print(*prompt)
	}
	start := time.Now()
	count := 0
	for t, err := range generate.Generate(ctx, m, tok, *prompt, cfg) {
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(t.Text) // streaming: text appears as soon as a character is complete
		count++
	}
	fmt.Fprintf(os.Stderr, "\n[%d tokens in %v, %.0f tokens/s]\n", count, time.Since(start).Round(time.Millisecond),
		float64(count)/time.Since(start).Seconds())
}

func speculative(ctx context.Context, target *gpt.Model, draftPath string, tok train.Tokenizer, prompt string,
	n, k int, sc generate.SamplerConfig) {
	dc, err := gpt.LoadCheckpointFile(draftPath)
	if err != nil {
		log.Fatal(err)
	}
	draft := dc.Model
	ids := tok.Encode(prompt)
	t0 := time.Now()
	plain, _, err := generate.IDs(ctx, target, tok, ids, generate.Config{MaxTokens: n, Sampler: sc})
	if err != nil {
		log.Fatal(err)
	}
	tPlain := time.Since(t0)
	out, st, err := generate.Speculative(ctx, target, draft, ids, k, n, sc)
	if err != nil {
		log.Fatal(err)
	}
	t1 := time.Now()
	generate.IDs(ctx, draft, tok, ids, generate.Config{MaxTokens: n, Sampler: sc})
	tDraft := time.Since(t1)
	fmt.Printf("%s%s\n\n", prompt, tok.Decode(out))
	fmt.Printf("target %s (%d params), draft %s (%d params), k=%d\n", target.Cfg.Name, target.NumParams(),
		draft.Cfg.Name, draft.NumParams(), k)
	fmt.Printf("acceptance α = %d/%d = %.3f; tokens per target pass %.2f (formula (1−α^(k+1))/(1−α) = %.2f)\n",
		st.Accepted, st.Proposed, st.AcceptRate, st.TokensRound, generate.ExpectedTokensPerRound(st.AcceptRate, k))
	fmt.Printf("time: target alone %v, draft alone %v, speculative %v → speedup %.2f×\n",
		tPlain.Round(time.Millisecond), tDraft.Round(time.Millisecond), st.Elapsed.Round(time.Millisecond),
		tPlain.Seconds()/st.Elapsed.Seconds())
	if sc.Temperature == 0 {
		same := len(plain) == len(out)
		for i := range min(len(plain), len(out)) {
			same = same && plain[i] == out[i]
		}
		fmt.Printf("greedy output identical to the target's: %v\n", same)
	}
}

// benchmark measures, for growing sequence lengths, the time per generated token with
// and without the KV-cache, and the prefill throughput.
func benchmark(m *gpt.Model) {
	ctx := context.Background()
	fmt.Printf("%8s %14s %14s %9s\n", "tokens", "no cache", "KV-cache", "speedup")
	for _, n := range []int{16, 32, 64, 128, 256, 512} {
		if n >= m.Cfg.Context {
			break
		}
		var per [2]time.Duration
		for i, noCache := range []bool{true, false} {
			cfg := generate.Config{MaxTokens: n, NoCache: noCache}
			best := time.Duration(1 << 62)
			for range 3 {
				t0 := time.Now()
				generate.IDs(ctx, m, byteless{m.Cfg.Vocab}, []int{0}, cfg)
				best = min(best, time.Since(t0))
			}
			per[i] = best
		}
		fmt.Printf("%8d %11.2f ms %11.2f ms %8.1f×\n", n, ms(per[0]), ms(per[1]), per[0].Seconds()/per[1].Seconds())
	}
	// Prefill vs decode: the same number of tokens in one pass vs one at a time.
	t := min(256, m.Cfg.Context-1)
	ids := make([]int, t)
	s := m.NewSession(1, 0)
	t0 := time.Now()
	s.Forward([][]int{ids})
	prefill := time.Since(t0)
	s.Reset()
	t0 = time.Now()
	for i := range t {
		s.Forward([][]int{ids[i : i+1]})
	}
	decode := time.Since(t0)
	fmt.Printf("prefill %d tokens in one pass: %.2f ms (%.0f tokens/s); one by one: %.2f ms (%.0f tokens/s)\n",
		t, ms(prefill), float64(t)/prefill.Seconds(), ms(decode), float64(t)/decode.Seconds())
}

func ms(d time.Duration) float64 { return float64(d.Microseconds()) / 1000 }

// byteless is a tokenizer stub for the benchmark: ids only, no text.
type byteless struct{ v int }

func (b byteless) Encode(string) []int   { return nil }
func (b byteless) TokenBytes(int) []byte { return nil }
func (b byteless) VocabSize() int        { return b.v }
