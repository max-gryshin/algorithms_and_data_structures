package rag

import (
	"context"
	"fmt"
	"strings"
	"sync"
)

// Result is a retrieved chunk with its fused score.
type Result struct {
	Chunk Chunk
	Score float64
}

// Store keeps chunks in two indexes — BM25 (exact words) and vectors (meaning) —
// and retrieves with hybrid search fused by RRF.
type Store struct {
	Embedder Embedder
	Vectors  VectorIndex
	BM25     *BM25
	RRFK     int // RRF constant, 60 by default

	mu     sync.RWMutex
	chunks []Chunk
}

// NewStore returns a store with BM25(k1=1.2, b=0.75) and the given vector index.
func NewStore(emb Embedder, vec VectorIndex) *Store {
	return &Store{Embedder: emb, Vectors: vec, BM25: NewBM25(1.2, 0.75), RRFK: 60}
}

// Add embeds and indexes chunks. The heading is indexed together with the text,
// a cheap form of "contextual retrieval".
func (s *Store) Add(ctx context.Context, chunks ...Chunk) error {
	texts := make([]string, len(chunks))
	for i, c := range chunks {
		texts[i] = c.Heading + "\n" + c.Text
	}
	vecs, err := s.Embedder.Embed(ctx, texts)
	if err != nil {
		return fmt.Errorf("rag: embed: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for i, c := range chunks {
		s.BM25.Add(texts[i])
		s.Vectors.Add(vecs[i])
		s.chunks = append(s.chunks, c)
	}
	return nil
}

func (s *Store) Len() int {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.chunks)
}

// Retrieve returns the k best chunks: top-(4k) from BM25 and from vectors, fused by RRF.
func (s *Store) Retrieve(ctx context.Context, query string, k int) ([]Result, error) {
	qv, err := s.Embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, fmt.Errorf("rag: embed query: %w", err)
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	pool := 4 * k
	fused := RRF(s.RRFK, s.BM25.Search(query, pool), s.Vectors.Search(qv[0], pool))
	if len(fused) > k {
		fused = fused[:k]
	}
	out := make([]Result, len(fused))
	for i, h := range fused {
		out[i] = Result{Chunk: s.chunks[h.Index], Score: h.Score}
	}
	return out, nil
}

// BuildPrompt formats the question and numbered sources so the model can cite them as [n].
func BuildPrompt(question string, sources []Result) string {
	var b strings.Builder
	b.WriteString("Ответь на вопрос, используя только источники ниже. ")
	b.WriteString("После каждого утверждения укажи номер источника в квадратных скобках, например [1]. ")
	b.WriteString("Если в источниках нет ответа, так и скажи.\n\n<sources>\n")
	for i, r := range sources {
		fmt.Fprintf(&b, "<source id=\"%d\" file=%q section=%q>\n%s\n</source>\n", i+1, r.Chunk.Source, r.Chunk.Heading, r.Chunk.Text)
	}
	fmt.Fprintf(&b, "</sources>\n\nВопрос: %s\n", question)
	return b.String()
}

// Answer is a generated answer together with the sources it was given.
type Answer struct {
	Text    string
	Prompt  string
	Sources []Result
}

// Pipeline is the full RAG loop: retrieve → build prompt → generate.
type Pipeline struct {
	Store     *Store
	Generator Generator
	K         int // number of chunks to put in the prompt
}

func (p *Pipeline) Answer(ctx context.Context, question string) (Answer, error) {
	sources, err := p.Store.Retrieve(ctx, question, p.K)
	if err != nil {
		return Answer{}, err
	}
	prompt := BuildPrompt(question, sources)
	text, err := p.Generator.Generate(ctx, prompt)
	if err != nil {
		return Answer{}, fmt.Errorf("rag: generate: %w", err)
	}
	return Answer{Text: text, Prompt: prompt, Sources: sources}, nil
}
