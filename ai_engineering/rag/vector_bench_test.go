package rag

import (
	"fmt"
	"math/rand/v2"
	"testing"
)

// clusteredVecs imitates real embeddings: points gather around a few hundred
// topics (low intrinsic dimension), unlike pure Gaussian noise.
func clusteredVecs(rng *rand.Rand, n, d int) [][]float32 {
	centers := randVecs(rng, 200, d)
	out := randVecs(rng, n, d)
	for i, v := range out {
		c := centers[rng.IntN(len(centers))]
		for j := range v {
			v[j] = c[j] + 0.3*v[j]
		}
		out[i] = v
	}
	return out
}

// go test ./ai_engineering/rag -run xxx -bench Search -benchtime 3000x
func BenchmarkSearch(b *testing.B) {
	const d, k = 64, 10
	for _, data := range []string{"gauss", "clustered"} {
		for _, n := range []int{10_000, 50_000} {
			rng := rand.New(rand.NewPCG(1, 1))
			gen := randVecs
			if data == "clustered" {
				gen = clusteredVecs
			}
			all := gen(rng, n+256, d)
			base, qs := all[:n], all[n:]
			f, g := &Flat{}, NewHNSW(16, 200, 64, 1)
			for _, v := range base {
				f.Add(v)
				g.Add(v)
			}
			b.Run(fmt.Sprintf("%s/flat/n=%d", data, n), func(b *testing.B) {
				i := 0
				for b.Loop() {
					f.Search(qs[i%len(qs)], k)
					i++
				}
			})
			for _, ef := range []int{32, 64, 256} {
				g.EfSearch = ef
				var recall float64
				for _, q := range qs {
					exact := map[int]bool{}
					for _, h := range f.Search(q, k) {
						exact[h.Index] = true
					}
					var got []int
					for _, h := range g.Search(q, k) {
						got = append(got, h.Index)
					}
					recall += RecallAtK(got, exact, k) / float64(len(qs))
				}
				b.Run(fmt.Sprintf("%s/hnsw-ef%d/n=%d", data, ef, n), func(b *testing.B) {
					i := 0
					for b.Loop() {
						g.Search(qs[i%len(qs)], k)
						i++
					}
					b.ReportMetric(recall, "recall@10")
				})
			}
		}
	}
}
