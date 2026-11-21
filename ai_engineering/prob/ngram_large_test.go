package prob

import (
	"flag"
	"math/rand/v2"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
	"unicode"
	"unicode/utf8"
)

// Flags of TestGenerateLargeCorpus. The test is opt-in: it is skipped unless -corpus is set.
//
//	go test ./prob -run TestGenerateLargeCorpus -v -timeout 60m -args \
//	  -corpus=../data/corpus_large.txt -n=3,5,8 -t=0,0.5,1 -prompt='Теорема|Инфляция' -len=300
//
// The path is relative to prob/, where go test runs. One model is trained per n and reused for
// every (t, prompt) pair.
var (
	flagCorpus = flag.String("corpus", "", "training text for TestGenerateLargeCorpus; empty = skip the test")
	flagN      = flag.String("n", "2,4,6,8", "comma-separated n-gram orders")
	flagT      = flag.String("t", "0,0.5,1", "comma-separated temperatures; 0 = greedy")
	flagPrompt = flag.String("prompt", "Теорема|Алгоритм|Нейронная сеть|Император|Инфляция",
		"prompts separated by |; pick words that occur in the corpus (math, CS, AI, history, economics)")
	flagLen  = flag.Int("len", 200, "characters to generate after the prompt")
	flagK    = flag.Float64("k", 0, "add-k smoothing")
	flagSeed = flag.Uint64("seed", 1, "sampling seed")
	flagRu   = flag.Bool("ru", true, "keep only mostly-Cyrillic lines and a small alphabet (V≈100); "+
		"without it V≈700 and n ≥ 6 needs tens of GB of RAM")
)

func TestGenerateLargeCorpus(t *testing.T) {
	if *flagCorpus == "" {
		t.Skip("set -corpus=<file> to run, see the comment on the flags")
	}
	b, err := os.ReadFile(*flagCorpus)
	if err != nil {
		t.Fatal(err)
	}
	text := string(b)
	if *flagRu {
		text = russianProse(text)
	}
	vocab := Vocab(text)
	t.Logf("corpus %s: %d chars, |V| = %d", *flagCorpus, utf8.RuneCountInString(text), len(vocab))

	prompts := strings.Split(*flagPrompt, "|")
	for _, n := range parseList(t, *flagN, strconv.Atoi) {
		start := time.Now()
		m := NewNGram(n, *flagK, vocab)
		m.Train(text)
		t.Logf("=== n = %d: %d contexts, trained in %s", n, len(m.totals), time.Since(start).Round(time.Second))
		for _, temp := range parseList(t, *flagT, func(s string) (float64, error) { return strconv.ParseFloat(s, 64) }) {
			for _, p := range prompts {
				out := m.Generate(rand.New(rand.NewPCG(*flagSeed, *flagSeed)), p, *flagLen, temp)
				t.Logf("n=%d t=%g %q\n%s\n", n, temp, p, out)
			}
		}
	}
}

// russianProse keeps the lines whose letters are mostly Cyrillic and drops every character outside
// Cyrillic letters, digits, space and basic punctuation. This keeps |V| near 100, so the dense
// per-context rows of NGram fit in memory up to n ≈ 8. Runs of spaces left by the dropped
// characters are collapsed, otherwise greedy decoding gets stuck on "space → space".
func russianProse(text string) string {
	var b strings.Builder
	for line := range strings.Lines(text) {
		var cyr, lat int
		for _, r := range line {
			switch {
			case unicode.Is(unicode.Cyrillic, r):
				cyr++
			case r < utf8.RuneSelf && unicode.IsLetter(r):
				lat++
			}
		}
		if cyr < 20 || cyr < 4*lat {
			continue
		}
		var kept strings.Builder
		for _, r := range line {
			if unicode.Is(unicode.Cyrillic, r) || strings.ContainsRune(" ,.:;!?-—()«»0123456789", r) {
				kept.WriteRune(r)
			}
		}
		b.WriteString(strings.Join(strings.Fields(kept.String()), " "))
		b.WriteByte('\n')
	}
	return b.String()
}

func parseList[T any](t *testing.T, s string, parse func(string) (T, error)) []T {
	t.Helper()
	var out []T
	for f := range strings.SplitSeq(s, ",") {
		v, err := parse(strings.TrimSpace(f))
		if err != nil {
			t.Fatalf("bad list %q: %v", s, err)
		}
		out = append(out, v)
	}
	return out
}
