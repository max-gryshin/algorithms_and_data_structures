package gpt

import (
	stdbytes "bytes" // calc.go declares a func bytes
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"algorithms_and_data_structures/ai_engineering/tensor"
)

// lastRows returns rows [from, to) along T of logits (B, T, V) as (B, to−from, V).
func lastRows(t *tensor.Tensor, from, to int) []float32 {
	s := t.Shape()
	b, tt, v := s[0], s[1], s[2]
	var out []float32
	for bi := range b {
		out = append(out, t.Data()[(bi*tt+from)*v:(bi*tt+to)*v]...)
	}
	return out
}

// The KV-cache must not change a single bit of the logits: prefill a prompt, then
// feed tokens one at a time and in chunks, and compare with a full forward each time.
func TestSessionMatchesForward(t *testing.T) {
	rng := rand.New(rand.NewPCG(10, 10))
	for _, cfg := range allConfigs(40, 24) {
		m := New(rng, cfg)
		ids := randIDs(rng, 2, 20, 40)
		s := m.NewSession(2, 0)
		feed := []int{7, 1, 1, 3, 1, 5, 2} // prefill 7, then single tokens and chunks: 7+1+1+3+1+5+2 = 20
		pos := 0
		for _, n := range feed {
			chunk := make([][]int, 2)
			for b := range chunk {
				chunk[b] = ids[b][pos : pos+n]
			}
			got := s.Forward(chunk)
			full := m.Forward(prefix(ids, pos+n)).Data
			if want := lastRows(full, pos, pos+n); !slices.Equal(got.Data(), want) {
				t.Fatalf("%s: logits of tokens %d..%d differ from the full forward", cfg.Name, pos, pos+n)
			}
			pos += n
		}
		if s.Len() != 20 {
			t.Fatalf("%s: session length %d", cfg.Name, s.Len())
		}
		// Truncate and replay: the same logits again.
		s.Truncate(10)
		got := s.Forward(prefixFrom(ids, 10, 12))
		if want := lastRows(m.Forward(prefix(ids, 12)).Data, 10, 12); !slices.Equal(got.Data(), want) {
			t.Fatalf("%s: logits after Truncate differ", cfg.Name)
		}
		s.Reset()
		if got := s.Forward(prefix(ids, 3)); !slices.Equal(got.Data(), m.Forward(prefix(ids, 3)).Data.Data()) {
			t.Fatalf("%s: logits after Reset differ", cfg.Name)
		}
	}
}

func prefix(ids [][]int, n int) [][]int { return prefixFrom(ids, 0, n) }

func prefixFrom(ids [][]int, from, to int) [][]int {
	out := make([][]int, len(ids))
	for i := range ids {
		out[i] = ids[i][from:to]
	}
	return out
}

// Memory of the cache: 2·L·B·h_kv·d_head·T·4 bytes.
func TestSessionBytes(t *testing.T) {
	cfg := TinyLlama(30, 16) // L=2, h_kv=2, d_head=8
	m := New(rand.New(rand.NewPCG(1, 1)), cfg)
	s := m.NewSession(3, 0)
	s.Forward(randIDs(rand.New(rand.NewPCG(2, 2)), 3, 5, 30))
	if got, want := s.Bytes(), 2*2*3*2*8*5*4; got != want {
		t.Fatalf("bytes %d, want %d", got, want)
	}
}

func TestCheckpointRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 11))
	for _, cfg := range allConfigs(30, 12) {
		cfg.Dropout = 0.1
		m := New(rng, cfg)
		var buf stdbytes.Buffer
		c := &Checkpoint{Model: m, Meta: map[string]string{"tokenizer": "char"},
			Extra: map[string][]float64{"adam.m.tok.W": {1, math.Pi, -2.5}}}
		if err := c.Save(&buf); err != nil {
			t.Fatal(err)
		}
		got, err := LoadCheckpoint(&buf)
		if err != nil {
			t.Fatal(err)
		}
		if got.Model.Cfg != m.Cfg || got.Meta["tokenizer"] != "char" || got.Extra["adam.m.tok.W"][1] != math.Pi {
			t.Fatalf("%s: config/meta/extra not restored: %+v", cfg.Name, got.Model.Cfg)
		}
		ids := randIDs(rng, 2, 6, 30)
		if !slices.Equal(got.Model.Forward(ids).Data.Data(), m.Forward(ids).Data.Data()) {
			t.Fatalf("%s: restored model gives other logits", cfg.Name)
		}
	}
	if _, err := LoadCheckpoint(stdbytes.NewReader([]byte("not a checkpoint at all"))); err == nil {
		t.Fatal("garbage accepted")
	}
}

func TestNamedParameters(t *testing.T) {
	for _, cfg := range allConfigs(30, 12) {
		m := New(rand.New(rand.NewPCG(1, 2)), cfg)
		named, plain := m.NamedParameters(), m.Parameters()
		if len(named) != len(plain) {
			t.Fatalf("%s: %d named vs %d params", cfg.Name, len(named), len(plain))
		}
		seen := map[string]bool{}
		for i, np := range named {
			if seen[np.Name] || np.P != plain[i] {
				t.Fatalf("%s: name %s duplicated or out of order", cfg.Name, np.Name)
			}
			seen[np.Name] = true
		}
	}
}

func TestReplicaSharesWeightsNotGrads(t *testing.T) {
	rng := rand.New(rand.NewPCG(12, 12))
	m := New(rng, TinyGPT2(20, 8))
	r := m.Replica()
	ids := randIDs(rng, 2, 8, 20)
	if !slices.Equal(r.Forward(ids).Data.Data(), m.Forward(ids).Data.Data()) {
		t.Fatal("replica computes something else")
	}
	r.Loss(ids, ids).Backward()
	for i, p := range m.Parameters() {
		if p.Grad != nil || r.Parameters()[i].Grad == nil {
			t.Fatal("gradients must stay in the replica")
		}
	}
	m.Tok.W.Data.Data()[0] += 1 // an optimizer step on the master is visible in the replica
	if r.Tok.W.Data.Data()[0] != m.Tok.W.Data.Data()[0] {
		t.Fatal("weights are not shared")
	}
}

// Dropout is active only in training mode, is deterministic for a seed, and keeps
// the expected activation (inverted dropout), so eval-mode loss is unaffected.
func TestDropoutTrainEval(t *testing.T) {
	cfg := TinyGPT2(20, 8)
	cfg.Dropout = 0.2
	m := New(rand.New(rand.NewPCG(13, 13)), cfg)
	ids := randIDs(rand.New(rand.NewPCG(14, 14)), 2, 8, 20)
	eval1 := m.Forward(ids).Data.Data()
	m.Train(rand.New(rand.NewPCG(1, 1)))
	tr1 := m.Forward(ids).Data.Data()
	m.Train(rand.New(rand.NewPCG(1, 1)))
	tr2 := m.Forward(ids).Data.Data()
	m.Eval()
	eval2 := m.Forward(ids).Data.Data()
	if slices.Equal(eval1, tr1) || !slices.Equal(tr1, tr2) || !slices.Equal(eval1, eval2) {
		t.Fatal("dropout must change training forward only, deterministically per seed")
	}
}
