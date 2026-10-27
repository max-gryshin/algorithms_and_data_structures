package rag

import (
	"context"
	"math"
	"math/rand/v2"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

func TestTokenizeRussian(t *testing.T) {
	got := strings.Join(Tokenize("Кот СПИТ, cat! d_k=128"), "|")
	if got != "кот|спит|cat|d|k|128" {
		t.Fatalf("got %q", got)
	}
}

// TestBM25HandExample reproduces the worked example from lecture L17, section 6.
func TestBM25HandExample(t *testing.T) {
	x := NewBM25(1.2, 0.75)
	x.Add("кот спит на диване")
	x.Add("собака спит")
	x.Add("кот ловит мышь кот")

	if !near(x.AvgLen(), 10.0/3, 1e-9) {
		t.Fatalf("avgdl = %v", x.AvgLen())
	}
	if !near(x.IDF("кот"), 0.4700, 1e-4) || !near(x.IDF("мышь"), 0.9808, 1e-4) {
		t.Fatalf("IDF кот=%.4f мышь=%.4f", x.IDF("кот"), x.IDF("мышь"))
	}
	hits := x.Search("кот мышь", 10)
	if len(hits) != 2 || hits[0].Index != 2 || hits[1].Index != 0 {
		t.Fatalf("hits = %+v", hits)
	}
	if !near(hits[0].Score, 1.5185, 1e-4) || !near(hits[1].Score, 0.4345, 1e-4) {
		t.Fatalf("scores = %+v", hits)
	}
	// Length normalization: for "спит" the short document wins.
	if s1, s2 := x.Score("спит", 0), x.Score("спит", 1); !(s2 > s1) || !near(s2, 0.5620, 1e-4) {
		t.Fatalf("спит: long=%.4f short=%.4f", s1, s2)
	}
}

func TestRRFHandExample(t *testing.T) {
	const A, B, C, D = 0, 1, 2, 3
	bm := []Hit{{A, 9}, {B, 7}, {C, 1}}
	vec := []Hit{{B, .9}, {D, .8}, {A, .7}}
	got := RRF(60, bm, vec)
	order := []int{B, A, D, C}
	for i, h := range got {
		if h.Index != order[i] {
			t.Fatalf("order = %+v, want %v", got, order)
		}
	}
	if !near(got[0].Score, 1.0/62+1.0/61, 1e-12) {
		t.Fatalf("score(B) = %v", got[0].Score)
	}
}

func TestMetricsHandExample(t *testing.T) {
	retrieved := []int{3, 1, 5}
	rel := map[int]bool{1: true, 2: true, 3: true}
	if r := RecallAtK(retrieved, rel, 3); !near(r, 2.0/3, 1e-12) {
		t.Fatalf("recall = %v", r)
	}
	if rr := ReciprocalRank([]int{7, 5, 1}, rel); !near(rr, 1.0/3, 1e-12) {
		t.Fatalf("rr = %v", rr)
	}
	if n := NDCGAtK(retrieved, map[int]float64{1: 3, 2: 2, 3: 1}, 3); !near(n, 0.6075, 1e-4) {
		t.Fatalf("ndcg = %v", n)
	}
}

func TestVectorMath(t *testing.T) {
	a, b := []float32{1, 2, 3}, []float32{4, -5, 6}
	if Dot(a, b) != 12 || !near(float64(Cosine(a, b)), 0.3655, 1e-4) {
		t.Fatalf("dot=%v cos=%v", Dot(a, b), Cosine(a, b))
	}
	if !near(float64(L2([]float32{0, 0}, []float32{3, 4})), 5, 1e-6) {
		t.Fatal("L2")
	}
}

func randVecs(rng *rand.Rand, n, d int) [][]float32 {
	out := make([][]float32, n)
	for i := range out {
		out[i] = make([]float32, d)
		for j := range out[i] {
			out[i][j] = float32(rng.NormFloat64())
		}
	}
	return out
}

func TestFlatExact(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 1))
	f := &Flat{}
	vs := randVecs(rng, 200, 8)
	for _, v := range vs {
		f.Add(v)
	}
	// The nearest neighbour of a stored vector is itself.
	for i := 0; i < 200; i += 17 {
		if h := f.Search(vs[i], 3); h[0].Index != i || !near(h[0].Score, 1, 1e-5) {
			t.Fatalf("query %d: %+v", i, h)
		}
	}
}

func TestHNSWRecall(t *testing.T) {
	rng := rand.New(rand.NewPCG(2, 2))
	const n, d, k, queries = 3000, 32, 10, 100
	g, f := NewHNSW(16, 100, 64, 7), &Flat{}
	for _, v := range randVecs(rng, n, d) {
		g.Add(v)
		f.Add(v)
	}
	var recall float64
	for _, q := range randVecs(rng, queries, d) {
		exact := map[int]bool{}
		for _, h := range f.Search(q, k) {
			exact[h.Index] = true
		}
		var got []int
		for _, h := range g.Search(q, k) {
			got = append(got, h.Index)
		}
		recall += RecallAtK(got, exact, k)
	}
	recall /= queries
	if recall < 0.9 {
		t.Fatalf("HNSW recall@10 = %.3f, want ≥ 0.9", recall)
	}
	t.Logf("recall@10 = %.3f, nodes per top level = %v", recall, g.Levels())
}

func TestChunkMarkdown(t *testing.T) {
	doc := "# L1\nintro\n## 2. Векторы\ntext a\n```bash\n# not a heading\n```\n### 2.1. Сложение\nsum\n## 3. Матрицы\n" +
		strings.Repeat("абвгд ", 50)
	chunks := ChunkMarkdown("l1.md", doc, 120, 20)
	var heads []string
	for _, c := range chunks {
		heads = append(heads, c.Heading)
		if n := len([]rune(c.Text)); n > 120 {
			t.Fatalf("chunk %s has %d runes", c.ID, n)
		}
	}
	want := []string{"L1", "L1 > 2. Векторы", "L1 > 2. Векторы > 2.1. Сложение", "L1 > 3. Матрицы"}
	for i, w := range want {
		if heads[i] != w {
			t.Fatalf("headings = %q", heads)
		}
	}
	if !strings.Contains(chunks[1].Text, "# not a heading") {
		t.Fatal("code fence must stay inside its section")
	}
	if len(chunks) < 6 {
		t.Fatalf("long section must be split, got %d chunks", len(chunks))
	}
}

type echoGenerator struct{ prompt string }

func (e *echoGenerator) Generate(_ context.Context, prompt string) (string, error) {
	e.prompt = prompt
	return "Делят на √d_k, чтобы дисперсия скоров была 1 [1].", nil
}

// TestRAGOverLectures indexes the course's own lectures and asks a question from L1.
func TestRAGOverLectures(t *testing.T) {
	files, _ := filepath.Glob("../lectures/*.md")
	if len(files) == 0 {
		t.Skip("no lectures on disk")
	}
	ctx := context.Background()
	store := NewStore(HashEmbedder{Dim: 512}, NewHNSW(16, 100, 64, 1))
	for _, f := range files {
		b, err := os.ReadFile(f)
		if err != nil {
			t.Fatal(err)
		}
		if err := store.Add(ctx, ChunkMarkdown(filepath.Base(f), string(b), 1500, 200)...); err != nil {
			t.Fatal(err)
		}
	}
	gen := &echoGenerator{}
	p := &Pipeline{Store: store, Generator: gen, K: 5}
	ans, err := p.Answer(ctx, "почему скоры делят на корень из размерности ключей √d_k")
	if err != nil {
		t.Fatal(err)
	}
	found := false
	for _, s := range ans.Sources {
		found = found || strings.Contains(s.Chunk.Text, "√d")
	}
	if !found {
		t.Fatalf("no chunk about √d in top-5: %+v", ans.Sources)
	}
	if !strings.Contains(gen.prompt, `<source id="1"`) || !strings.Contains(gen.prompt, "Вопрос:") {
		t.Fatal("prompt must contain numbered sources and the question")
	}
	t.Logf("%d chunks indexed; top source: %s / %s", store.Len(), ans.Sources[0].Chunk.Source, ans.Sources[0].Chunk.Heading)
}
