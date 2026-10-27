package scaling

import (
	"hash/fnv"
	"math"
	"math/bits"
	"math/rand/v2"
	"slices"
	"strings"
)

// Shingles returns the set of word k-grams of text (lower-cased, split on whitespace).
func Shingles(text string, k int) map[string]struct{} {
	words := strings.Fields(strings.ToLower(text))
	set := make(map[string]struct{})
	for i := 0; i+k <= len(words); i++ {
		set[strings.Join(words[i:i+k], " ")] = struct{}{}
	}
	if len(words) > 0 && len(words) < k {
		set[strings.Join(words, " ")] = struct{}{}
	}
	return set
}

// Jaccard returns |A ∩ B| / |A ∪ B| (1 for two empty sets).
func Jaccard(a, b map[string]struct{}) float64 {
	if len(a) == 0 && len(b) == 0 {
		return 1
	}
	inter := 0
	for s := range a {
		if _, ok := b[s]; ok {
			inter++
		}
	}
	return float64(inter) / float64(len(a)+len(b)-inter)
}

// mersenne61 is the prime 2^61 − 1 used for universal hashing h(x) = (a·x + b) mod p.
const mersenne61 = 1<<61 - 1

// MinHasher computes MinHash signatures with k independent hash functions.
type MinHasher struct{ a, b []uint64 }

// NewMinHasher draws k random hash functions from a seeded generator.
func NewMinHasher(k int, seed uint64) *MinHasher {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	m := &MinHasher{a: make([]uint64, k), b: make([]uint64, k)}
	for i := range k {
		m.a[i] = rng.Uint64N(mersenne61-1) + 1
		m.b[i] = rng.Uint64N(mersenne61)
	}
	return m
}

// Signature returns, for every hash function, the minimum hash over the shingle set.
// For two sets the probability that their i-th minimums coincide equals their Jaccard similarity.
func (m *MinHasher) Signature(shingles map[string]struct{}) []uint64 {
	sig := make([]uint64, len(m.a))
	for i := range sig {
		sig[i] = math.MaxUint64
	}
	for s := range shingles {
		h := fnv.New64a()
		h.Write([]byte(s))
		x := h.Sum64() % mersenne61
		for i := range sig {
			if v := mulAddMod(m.a[i], x, m.b[i]); v < sig[i] {
				sig[i] = v
			}
		}
	}
	return sig
}

// mulAddMod returns (a·x + b) mod 2^61−1 without overflow, using 128-bit multiplication.
func mulAddMod(a, x, b uint64) uint64 {
	hi, lo := bits.Mul64(a, x)
	// a·x = hi·2^64 + lo; with p = 2^61−1, 2^61 ≡ 1, so fold the bits above 61 back in.
	r := (lo & mersenne61) + (lo >> 61) + (hi << 3)
	r = (r & mersenne61) + (r >> 61)
	r += b
	r = (r & mersenne61) + (r >> 61)
	if r >= mersenne61 {
		r -= mersenne61
	}
	return r
}

// EstimateJaccard is the fraction of positions where two signatures agree.
func EstimateJaccard(s1, s2 []uint64) float64 {
	eq := 0
	for i := range s1 {
		if s1[i] == s2[i] {
			eq++
		}
	}
	return float64(eq) / float64(len(s1))
}

// LSHProbability is the chance two documents with Jaccard similarity s become a candidate pair
// when a signature is split into b bands of r rows: 1 − (1 − s^r)^b.
func LSHProbability(s float64, bands, rows int) float64 {
	return 1 - math.Pow(1-math.Pow(s, float64(rows)), float64(bands))
}

// NearDuplicates returns index pairs (i < j) of documents whose estimated Jaccard similarity
// is at least threshold. Candidates come from LSH banding, so only documents sharing at least
// one identical band are compared — far fewer than all n·(n−1)/2 pairs.
func NearDuplicates(docs []string, shingleK, bands, rows int, threshold float64, seed uint64) [][2]int {
	mh := NewMinHasher(bands*rows, seed)
	sigs := make([][]uint64, len(docs))
	for i, d := range docs {
		sigs[i] = mh.Signature(Shingles(d, shingleK))
	}

	candidates := make(map[[2]int]struct{})
	for band := range bands {
		buckets := make(map[string][]int)
		for i, sig := range sigs {
			key := bandKey(sig[band*rows : (band+1)*rows])
			buckets[key] = append(buckets[key], i)
		}
		for _, ids := range buckets {
			for x := 0; x < len(ids); x++ {
				for y := x + 1; y < len(ids); y++ {
					candidates[[2]int{ids[x], ids[y]}] = struct{}{}
				}
			}
		}
	}

	var out [][2]int
	for p := range candidates {
		if EstimateJaccard(sigs[p[0]], sigs[p[1]]) >= threshold {
			out = append(out, p)
		}
	}
	slices.SortFunc(out, func(a, b [2]int) int {
		if a[0] != b[0] {
			return a[0] - b[0]
		}
		return a[1] - b[1]
	})
	return out
}

func bandKey(rows []uint64) string {
	var sb strings.Builder
	for _, v := range rows {
		for s := 0; s < 64; s += 8 {
			sb.WriteByte(byte(v >> s))
		}
	}
	return sb.String()
}
