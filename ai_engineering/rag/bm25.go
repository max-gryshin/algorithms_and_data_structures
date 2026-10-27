package rag

import (
	"math"
	"sort"
)

// Hit is one search result: the index of the document (in insertion order) and its score.
// Higher score = more relevant, for every index in this package.
type Hit struct {
	Index int
	Score float64
}

// BM25 is an in-memory inverted index scored with Okapi BM25:
//
//	score(D, Q) = Σ_{t∈Q} IDF(t) · tf·(k1+1) / (tf + k1·(1 − b + b·|D|/avgdl))
type BM25 struct {
	K1, B    float64
	tf       []map[string]int
	docLen   []int
	postings map[string][]int // term → documents containing it (the "inverted index")
	totalLen int
}

// NewBM25 returns an empty index. Typical parameters: k1 = 1.2, b = 0.75.
func NewBM25(k1, b float64) *BM25 {
	return &BM25{K1: k1, B: b, postings: map[string][]int{}}
}

// Add indexes a document and returns its index.
func (x *BM25) Add(text string) int {
	toks := Tokenize(text)
	tf := map[string]int{}
	for _, t := range toks {
		tf[t]++
	}
	i := len(x.tf)
	for t := range tf {
		x.postings[t] = append(x.postings[t], i)
	}
	x.tf = append(x.tf, tf)
	x.docLen = append(x.docLen, len(toks))
	x.totalLen += len(toks)
	return i
}

func (x *BM25) Len() int { return len(x.tf) }

// AvgLen is avgdl: the mean document length in tokens.
func (x *BM25) AvgLen() float64 {
	if len(x.tf) == 0 {
		return 0
	}
	return float64(x.totalLen) / float64(len(x.tf))
}

// IDF = ln((N − df + 0.5)/(df + 0.5) + 1): rare terms weigh more; never negative.
func (x *BM25) IDF(term string) float64 {
	n, df := float64(len(x.tf)), float64(len(x.postings[term]))
	return math.Log((n-df+0.5)/(df+0.5) + 1)
}

// Score returns the BM25 score of document i for the query.
func (x *BM25) Score(query string, i int) float64 {
	norm := 1 - x.B + x.B*float64(x.docLen[i])/x.AvgLen()
	var s float64
	for _, t := range Tokenize(query) {
		tf := float64(x.tf[i][t])
		if tf == 0 {
			continue
		}
		s += x.IDF(t) * tf * (x.K1 + 1) / (tf + x.K1*norm)
	}
	return s
}

// Search returns up to k documents with a positive score, best first.
// It visits only documents that contain at least one query term.
func (x *BM25) Search(query string, k int) []Hit {
	seen := map[int]bool{}
	var hits []Hit
	for _, t := range Tokenize(query) {
		for _, i := range x.postings[t] {
			if !seen[i] {
				seen[i] = true
				hits = append(hits, Hit{Index: i, Score: x.Score(query, i)})
			}
		}
	}
	sortHits(hits)
	if len(hits) > k {
		hits = hits[:k]
	}
	return hits
}

func sortHits(h []Hit) {
	sort.Slice(h, func(i, j int) bool {
		if h[i].Score != h[j].Score {
			return h[i].Score > h[j].Score
		}
		return h[i].Index < h[j].Index
	})
}

// RRF fuses several ranked lists with Reciprocal Rank Fusion:
//
//	score(d) = Σ_lists 1 / (k + rank(d)),  rank starting at 1.
//
// Only ranks matter, so lists with incomparable scores (BM25 vs cosine) can be mixed.
// k = 60 is the value from the original paper.
func RRF(k int, lists ...[]Hit) []Hit {
	acc := map[int]float64{}
	for _, l := range lists {
		for r, h := range l {
			acc[h.Index] += 1 / float64(k+r+1)
		}
	}
	out := make([]Hit, 0, len(acc))
	for i, s := range acc {
		out = append(out, Hit{Index: i, Score: s})
	}
	sortHits(out)
	return out
}
