package prob

import (
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"testing"
	"unicode/utf8"
)

func loadCorpus(t testing.TB) (train, valid string, vocab []rune) {
	t.Helper()
	b, err := os.ReadFile("../data/corpus_ru.txt")
	if err != nil {
		t.Fatal(err)
	}
	text := string(b)
	runes := []rune(text)
	cut := len(runes) * 9 / 10
	return string(runes[:cut]), string(runes[cut:]), Vocab(text)
}

func TestBigramHandCounts(t *testing.T) {
	// "мама мыла раму": after 'м' come а, а, ы, у → p(а|м) = 2/4.
	text := "мама мыла раму"
	m := NewNGram(2, 0, Vocab(text))
	m.Train(text)
	if p := m.Prob("м", 'а'); p != 0.5 {
		t.Fatalf("p(а|м) = %v, want 0.5", p)
	}
	if p := m.Prob("ам", 'ы'); p != 0.25 { // only the last character matters for a bigram
		t.Fatalf("p(ы|м) = %v, want 0.25", p)
	}
	if p := m.Prob("м", 'р'); p != 0 {
		t.Fatalf("unseen bigram must be 0 without smoothing, got %v", p)
	}
	// Laplace (k=1), |V| = 7 (м а ы л р у and space): p(р|м) = (0+1)/(4+7) = 1/11.
	l := NewNGram(2, 1, Vocab(text))
	l.Train(text)
	if p := l.Prob("м", 'р'); !near(p, 1.0/11, 1e-12) {
		t.Fatalf("Laplace p(р|м) = %v, want 1/11", p)
	}
	var s float64
	for _, p := range l.Dist("м") {
		s += p
	}
	if !near(s, 1, 1e-12) {
		t.Fatalf("smoothed distribution sums to %v", s)
	}
}

func TestBigramBeatsUnigramOnHeldOut(t *testing.T) {
	train, valid, vocab := loadCorpus(t)
	uni := NewNGram(1, 1, vocab)
	uni.Train(train)
	bi := NewNGram(2, 1, vocab)
	bi.Train(train)
	pu, pb := uni.Perplexity(valid), bi.Perplexity(valid)
	t.Logf("|V|=%d train=%d valid=%d chars; perplexity: uniform=%d unigram=%.2f bigram=%.2f",
		len(vocab), utf8.RuneCountInString(train), utf8.RuneCountInString(valid), len(vocab), pu, pb)
	if !(pb < pu && pu < float64(len(vocab))) {
		t.Fatalf("want bigram < unigram < |V|: %.2f %.2f %d", pb, pu, len(vocab))
	}
	// Unsmoothed bigram assigns 0 to some held-out bigram → infinite perplexity.
	raw := NewNGram(2, 0, vocab)
	raw.Train(train)
	if !math.IsInf(raw.Perplexity(valid), 1) {
		t.Log("note: every held-out bigram was seen in training")
	}
}

func TestGenerateDeterministic(t *testing.T) {
	train, _, vocab := loadCorpus(t)
	m := NewNGram(2, 0, vocab)
	m.Train(train)
	gen := func() string { return m.Generate(rand.New(rand.NewPCG(1, 1)), "Кот", 80, 1) }
	a, b := gen(), gen()
	if a != b {
		t.Fatal("same seed must give the same text")
	}
	if !strings.HasPrefix(a, "Кот") || utf8.RuneCountInString(a) != 83 {
		t.Fatalf("unexpected generation %q", a)
	}
	// With k = 0 every generated bigram must exist in the training text.
	r := []rune(a)
	for i := 1; i < len(r); i++ {
		if !strings.Contains(train, string(r[i-1:i+1])) {
			t.Fatalf("generated unseen bigram %q", string(r[i-1:i+1]))
		}
	}
	t.Logf("sample: %q", a)
	greedy := m.Generate(rand.New(rand.NewPCG(1, 1)), "Кот", 40, 0)
	t.Logf("greedy: %q", greedy)
}
