package generate

import (
	"context"
	"fmt"
	"math/rand/v2"
	"testing"

	"algorithms_and_data_structures/ai_engineering/gpt"
)

// benchModel is a randomly initialized model of a realistic small shape: timing does
// not depend on the weights. d = 128, L = 4, h = 4, context 1024 (≈ 0.9M parameters).
func benchModel() *gpt.Model {
	cfg := gpt.TinyGPT2(256, 1024)
	cfg.DModel, cfg.Layers, cfg.Heads, cfg.KVHeads, cfg.FFHidden = 128, 4, 4, 4, 512
	return gpt.New(rand.New(rand.NewPCG(1, 1)), cfg)
}

// BenchmarkGenerate: time to generate T tokens greedily from a 1-token prompt, with
// the KV-cache (O(T) work per token… O(T²) total for attention only) and without it
// (the whole prefix is recomputed every step: O(T²) matmul work in total).
//
//	go test ./ai_engineering/gpt/generate -run '^$' -bench Generate -benchtime 3x
func BenchmarkGenerate(b *testing.B) {
	m := benchModel()
	for _, n := range []int{64, 128, 256, 512} {
		for _, noCache := range []bool{false, true} {
			name := fmt.Sprintf("T=%d/cache", n)
			if noCache {
				name = fmt.Sprintf("T=%d/nocache", n)
			}
			b.Run(name, func(b *testing.B) {
				cfg := Config{MaxTokens: n, NoCache: noCache}
				for b.Loop() {
					IDs(context.Background(), m, byteTok{}, []int{1}, cfg)
				}
				b.ReportMetric(float64(n)*float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
			})
		}
	}
}

// BenchmarkPrefillVsDecode: the same 256 tokens fed in one pass (prefill) or one at
// a time (decode). Prefill multiplies matrices by 256-row blocks; decode does 256
// matrix-vector products and rereads every weight 256 times.
func BenchmarkPrefillVsDecode(b *testing.B) {
	m := benchModel()
	ids := make([]int, 256)
	for i := range ids {
		ids[i] = i
	}
	s := m.NewSession(1, 0)
	b.Run("prefill", func(b *testing.B) {
		for b.Loop() {
			s.Reset()
			s.Forward([][]int{ids})
		}
		b.ReportMetric(256*float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
	})
	b.Run("decode", func(b *testing.B) {
		for b.Loop() {
			s.Reset()
			for i := range ids {
				s.Forward([][]int{ids[i : i+1]})
			}
		}
		b.ReportMetric(256*float64(b.N)/b.Elapsed().Seconds(), "tokens/s")
	})
}

// BenchmarkVerify: one target pass over k+1 new tokens after a 200-token context, the
// verification step of speculative decoding, compared with a single decode step.
func BenchmarkVerify(b *testing.B) {
	m := benchModel()
	ctx := make([]int, 200)
	s := m.NewSession(1, 0)
	s.Forward([][]int{ctx})
	for _, n := range []int{1, 2, 3, 5, 9} {
		b.Run(fmt.Sprintf("tokens=%d", n), func(b *testing.B) {
			ids := make([]int, n)
			for b.Loop() {
				s.Forward([][]int{ids})
				s.Truncate(200)
			}
		})
	}
}
