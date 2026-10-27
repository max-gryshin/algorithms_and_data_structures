package serve

import (
	"context"
	"math"
	"math/rand/v2"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/gpt/generate"
	"algorithms_and_data_structures/ai_engineering/gpt/train"
	"algorithms_and_data_structures/ai_engineering/llm"
)

func randn(rng *rand.Rand, n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = float32(rng.NormFloat64())
	}
	return x
}

func TestFlashEqualsNaive(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, c := range []struct {
		tq, tk, d, bq, bk int
		causal            bool
	}{{37, 37, 16, 8, 8, true}, {5, 64, 8, 4, 16, true}, {33, 50, 8, 7, 9, false}, {1, 100, 32, 1, 16, true}} {
		q, k, v := randn(rng, c.tq*c.d), randn(rng, c.tk*c.d), randn(rng, c.tk*c.d)
		want, _ := NaiveAttention(q, k, v, c.tq, c.tk, c.d, c.causal)
		got, _ := FlashAttention(q, k, v, c.tq, c.tk, c.d, c.bq, c.bk, c.causal)
		for i := range want {
			if math.Abs(float64(want[i]-got[i])) > 1e-5 {
				t.Fatalf("%+v: out[%d] = %v, naive %v", c, i, got[i], want[i])
			}
		}
	}
	// Scratch memory: T² for naive, constant for flash.
	for _, T := range []int{128, 1024} {
		q := randn(rng, T*16)
		_, ns := NaiveAttention(q, q, q, T, T, 16, true)
		_, fs := FlashAttention(q, q, q, T, T, 16, 32, 32, true)
		if ns != T*T || fs != 32*32+18*32 {
			t.Fatalf("T=%d: scratch naive %d, flash %d", T, ns, fs)
		}
	}
}

// The two-block example of the lecture: logits [1 3 | 2 5], values [10 20 | 30 40].
func TestOnlineSoftmaxTwoBlocks(t *testing.T) {
	s := NewOnlineSoftmax()
	if a := s.Push([]float64{1, 3}); a != 0 || s.M != 3 || math.Abs(s.L-1.135335) > 1e-6 {
		t.Fatalf("block 1: α=%v %+v", a, s)
	}
	if a := s.Push([]float64{2, 5}); math.Abs(a-math.Exp(-2)) > 1e-12 || math.Abs(s.L-1.203438) > 1e-6 {
		t.Fatalf("block 2: α=%v %+v", a, s)
	}
	b1, b2 := NewOnlineSoftmax(), NewOnlineSoftmax()
	b1.Push([]float64{1, 3})
	b2.Push([]float64{2, 5})
	if m := b1.Merge(b2); math.Abs(m.L-s.L) > 1e-12 || m.M != 5 {
		t.Fatalf("merge %+v", m)
	}
	out, _ := FlashAttention([]float32{1}, []float32{1, 3, 2, 5}, []float32{10, 20, 30, 40}, 1, 4, 1, 1, 2, false)
	if math.Abs(float64(out[0])-36.8806) > 1e-3 {
		t.Fatalf("out %v", out[0])
	}
}

func TestAllocatorAndFragmentation(t *testing.T) {
	a := NewBlockAllocator(4, 16)
	ids, ok := a.Alloc(3)
	if !ok || a.Free() != 1 {
		t.Fatal("alloc")
	}
	if _, ok := a.Alloc(2); ok {
		t.Fatal("alloc beyond capacity must fail atomically")
	}
	a.Retain(ids[0])
	a.Release(ids...)
	if a.Free() != 3 || a.RefCount(ids[0]) != 1 {
		t.Fatalf("free %d ref %d", a.Free(), a.RefCount(ids[0]))
	}
	f := Fragmentation([]int{10, 100, 1000}, 16, 2048)
	if f.Tokens != 1110 || f.PagedSlots != 16+112+1008 || math.Abs(f.ContigWaste-(1-1110.0/6144)) > 1e-12 {
		t.Fatalf("%+v", f)
	}
}

func TestRadixCache(t *testing.T) {
	a := NewBlockAllocator(16, 2)
	c := NewRadixCache(a)
	sys := []int{1, 2, 3, 4, 5, 6}
	b1, _ := a.Alloc(4)
	c.Insert(append(slices.Clone(sys), 7, 8, 9), b1) // 9 tokens → 4 full blocks
	a.Release(b1...)                                 // the sequence is done; only the cache holds them
	n, blocks := c.Match([]int{1, 2, 3, 4, 5, 6, 9, 9})
	if n != 6 || !slices.Equal(blocks, b1[:3]) {
		t.Fatalf("match %d %v", n, blocks)
	}
	b2, _ := a.Alloc(1)
	c.Insert([]int{1, 2, 3, 4, 5, 6, 9, 9}, append(slices.Clone(blocks), b2...)) // splits the edge after 6
	a.Release(b2...)
	if c.CachedBlocks() != 5 {
		t.Fatalf("cached %d", c.CachedBlocks())
	}
	if c.Evict(10) != 2 { // the two leaves; the shared prefix is pinned by our match
		t.Fatalf("evicted wrong amount, cached %d", c.CachedBlocks())
	}
	a.Release(blocks...)
	if c.Evict(10) != 3 || a.Free() != 16 {
		t.Fatalf("free %d", a.Free())
	}
}

func TestSplitStop(t *testing.T) {
	if e, s, h := splitStop("hello wo", []string{"world"}); e != "hello " || s || h != "wo" {
		t.Fatal(e, s, h)
	}
	if e, s, _ := splitStop("hello world!", []string{"world"}); e != "hello " || !s {
		t.Fatal(e, s)
	}
	if p := Percentile([]float64{4, 1, 3, 2}, 0.5); p != 2.5 {
		t.Fatal(p)
	}
}

func testModel(t *testing.T) (*gpt.Model, generate.Tokenizer) {
	tok := train.NewCharTokenizer("abcdefghijklmnopqrstuvwxyz :\n")
	m := gpt.New(rand.New(rand.NewPCG(3, 4)), gpt.TinyLlama(tok.VocabSize(), 96))
	return m, tok
}

// Under continuous batching, with chunked prefill and preemption, every request must get
// exactly the tokens it would get alone (greedy decoding).
func TestEngineMatchesGenerate(t *testing.T) {
	m, tok := testModel(t)
	for _, pol := range []Policy{Continuous, Static} {
		e := NewEngine(m, EngineConfig{Policy: pol, BlockSize: 4, NumBlocks: 40, MaxSeqs: 4, MaxBatchTokens: 16, PrefillChunk: 5})
		ctx, cancel := context.WithCancel(context.Background())
		go e.Run(ctx)
		var wg sync.WaitGroup
		for i := range 8 {
			wg.Go(func() {
				prompt := tok.Encode(strings.Repeat("ab cd", i+1))
				want, _, _ := generate.IDs(context.Background(), m, tok, prompt, generate.Config{MaxTokens: 10 + i})
				ch, err := e.Submit(context.Background(), Request{Prompt: prompt, MaxTokens: 10 + i})
				if err != nil {
					t.Error(err)
					return
				}
				var got []int
				for ev := range ch {
					if ev.Finish == "" {
						got = append(got, ev.Token)
					} else if ev.Finish != "length" || ev.CompletionTokens != 10+i {
						t.Errorf("finish %+v", ev)
					}
				}
				if !slices.Equal(got, want) {
					t.Errorf("policy %d request %d: %v, alone %v", pol, i, got, want)
				}
			})
		}
		wg.Wait()
		if st := e.Stats(); st.FreeBlocks != 40 || st.Running != 0 {
			t.Fatalf("policy %d: leaked blocks: %+v", pol, st)
		}
		cancel()
	}
}

func TestEngineCancellation(t *testing.T) {
	m, tok := testModel(t)
	e := NewEngine(m, EngineConfig{BlockSize: 4, NumBlocks: 64})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go e.Run(ctx)
	rctx, rcancel := context.WithCancel(context.Background())
	ch, err := e.Submit(rctx, Request{Prompt: tok.Encode("abc"), MaxTokens: 80})
	if err != nil {
		t.Fatal(err)
	}
	<-ch
	rcancel()
	var last Event
	for ev := range ch {
		last = ev
	}
	if last.Finish != "cancelled" {
		t.Fatalf("finish %q", last.Finish)
	}
	deadline := time.Now().Add(5 * time.Second)
	for e.Stats().FreeBlocks != 64 && time.Now().Before(deadline) {
		time.Sleep(time.Millisecond)
	}
	if e.Stats().FreeBlocks != 64 {
		t.Fatalf("blocks not freed: %+v", e.Stats())
	}
}

// The L16 client talks to our server: streaming and non-streaming.
func TestServerWithLLMClient(t *testing.T) {
	m, tok := testModel(t)
	e := NewEngine(m, EngineConfig{BlockSize: 4, NumBlocks: 64})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go e.Run(ctx)
	srv := httptest.NewServer((&Server{Engine: e, Tok: tok, Model: "gogpt"}).Handler())
	defer srv.Close()
	c := &llm.OpenAICompatible{BaseURL: srv.URL}
	zero := 0.0
	req := llm.Request{Model: "gogpt", System: "be brief", MaxTokens: 12, Temperature: &zero,
		Messages: []llm.Message{{Role: llm.RoleUser, Content: "hello"}}}
	full, err := c.Complete(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	if full.Usage.OutputTokens != 12 || full.StopReason != "length" || full.Usage.InputTokens == 0 {
		t.Fatalf("%+v", full)
	}
	streamed, err := llm.Collect(c.Stream(context.Background(), req))
	if err != nil {
		t.Fatal(err)
	}
	if streamed.Text != full.Text || streamed.Usage != full.Usage || streamed.StopReason != "length" {
		t.Fatalf("stream %+v vs %+v", streamed, full)
	}
	req.StopSequences = []string{full.Text[3:5]}
	stopped, err := c.Complete(context.Background(), req)
	if err != nil || stopped.StopReason != "stop" || !strings.HasPrefix(full.Text, stopped.Text) {
		t.Fatalf("stop: %+v %v", stopped, err)
	}
}
