// Package rag implements retrieval-augmented generation from scratch (lecture L17):
// lexical search (BM25), vector search (brute force and HNSW), hybrid fusion (RRF),
// Markdown chunking, retrieval metrics and a retrieve → prompt → generate pipeline.
package rag

import (
	"context"
	"hash/fnv"
	"math"
	"strings"
	"unicode"
)

// Tokenize lowercases s and splits it into runs of letters and digits.
// Works for any Unicode script, so "Кот СПИТ, cat!" → [кот спит cat].
func Tokenize(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
}

// Embedder turns texts into vectors. Real systems call an embedding model;
// tests use HashEmbedder so everything runs offline.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Generator produces an answer for a prompt (an LLM behind any API).
type Generator interface {
	Generate(ctx context.Context, prompt string) (string, error)
}

// HashEmbedder is a deterministic bag-of-words embedder ("hashing trick"):
// every word and every character trigram of a word is hashed into one of Dim buckets
// with a ±1 sign; the result is L2-normalized. It has no notion of meaning, but
// shares trigrams between word forms ("градиент" / "градиента"), which is enough
// to demonstrate dense retrieval without a neural model.
type HashEmbedder struct {
	Dim int
}

func (h HashEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = h.embed(t)
	}
	return out, nil
}

func (h HashEmbedder) embed(text string) []float32 {
	v := make([]float32, h.Dim)
	add := func(feature string, w float32) {
		f := fnv.New64a()
		f.Write([]byte(feature))
		x := f.Sum64()
		sign := float32(1)
		if x>>63 == 1 {
			sign = -1
		}
		v[x%uint64(h.Dim)] += sign * w
	}
	for _, tok := range Tokenize(text) {
		add("w:"+tok, 1)
		r := []rune("^" + tok + "$")
		for i := 0; i+3 <= len(r); i++ {
			add("g:"+string(r[i:i+3]), 0.5)
		}
	}
	Normalize(v)
	return v
}

// Dot returns Σ aᵢbᵢ.
func Dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

// Norm returns the Euclidean length √(Σ aᵢ²).
func Norm(a []float32) float32 { return float32(math.Sqrt(float64(Dot(a, a)))) }

// Cosine returns a·b / (‖a‖‖b‖), or 0 if either vector is zero.
func Cosine(a, b []float32) float32 {
	na, nb := Norm(a), Norm(b)
	if na == 0 || nb == 0 {
		return 0
	}
	return Dot(a, b) / (na * nb)
}

// L2 returns the Euclidean distance ‖a − b‖.
func L2(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return float32(math.Sqrt(float64(s)))
}

// Normalize scales v in place to unit length (no-op for the zero vector).
func Normalize(v []float32) {
	n := Norm(v)
	if n == 0 {
		return
	}
	for i := range v {
		v[i] /= n
	}
}
