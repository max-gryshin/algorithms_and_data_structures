package tokenizer

import (
	"bytes"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"unicode"
)

func mustTrain(t testing.TB, text string, vocabSize int, opts Options) *Tokenizer {
	t.Helper()
	tok, err := Train(text, vocabSize, opts)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func readData(t testing.TB, name string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("..", "data", name))
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestHandTraceASCII is the classic example traced by hand in the lecture (section 5.2).
func TestHandTraceASCII(t *testing.T) {
	tok := mustTrain(t, "aaabdaaabac", 259, Options{Pattern: PatternNone})
	want := []Pair{{'a', 'a'}, {'a', 'b'}, {256, 257}}
	if got := tok.Merges(); !slices.Equal(got, want) {
		t.Fatalf("merges = %v, want %v", got, want)
	}
	if got := tok.Encode("aaabdaaabac"); !slices.Equal(got, []int{258, 'd', 258, 'a', 'c'}) {
		t.Fatalf("encode = %v", got)
	}
	// Encoding new text applies merges by rank: "abaab" -> ab a ab -> ... "aa" (rank 256) wins first.
	if got := tok.Encode("abaab"); !slices.Equal(got, []int{257, 256, 'b'}) {
		t.Fatalf("encode(abaab) = %v", got)
	}
}

// TestHandTraceRussian is the Cyrillic example from the lecture (section 5.4):
// merges first glue bytes into letters, and some tokens end in the middle of a letter.
func TestHandTraceRussian(t *testing.T) {
	tok := mustTrain(t, "кот кот кот котик кит", 262, Options{Pattern: PatternGPT4})
	want := []Pair{{208, 186}, {209, 130}, {256, 208}, {' ', 258}, {190, 257}, {259, 260}}
	if got := tok.Merges(); !slices.Equal(got, want) {
		t.Fatalf("merges = %v, want %v", got, want)
	}
	if got := tok.Encode(" котик"); !slices.Equal(got, []int{261, 208, 184, 256}) {
		t.Fatalf("encode = %v", got)
	}
}

func TestSplitPatterns(t *testing.T) {
	text := "Hello world!!  How's it   going?\n\n  Цена 12345 руб."
	cases := []struct {
		pattern string
		want    []string
	}{
		{PatternGPT2, []string{"Hello", " world", "!!", " ", " How", "'s", " it", "  ", " going", "?", "\n\n ", " Цена", " 12345", " руб", "."}},
		{PatternGPT4, []string{"Hello", " world", "!!", " ", " How", "'s", " it", "  ", " going", "?\n\n", " ", " Цена", " ", "123", "45", " руб", "."}},
	}
	for _, c := range cases {
		s, err := newSplitter(c.pattern)
		if err != nil {
			t.Fatal(err)
		}
		if got := s.split(text); !slices.Equal(got, c.want) {
			t.Errorf("split = %q\nwant    %q", got, c.want)
		}
	}
}

var (
	sharedOnce sync.Once
	sharedTok  *Tokenizer
)

// shared returns a tokenizer trained once on the course corpus (used by round-trip and fuzz tests).
func shared(t testing.TB) *Tokenizer {
	sharedOnce.Do(func() {
		sharedTok = mustTrain(t, readData(t, "corpus_ru.txt")+readData(t, "parallel_en.txt"), 512,
			Options{Pattern: PatternGPT4, SpecialTokens: []string{"<|endoftext|>", "<|user|>"}})
	})
	return sharedTok
}

var roundTripCases = []string{
	"",
	"Привет, мир!",
	"Hello, world!",
	"emoji 😀👍🏽 and flags 🇺🇦",
	"invalid utf-8: \xff\xfe\xc3",
	"mixed\ttabs\r\nand   spaces   ",
	"func main() { fmt.Println(42) }",
	"<|endoftext|> must stay text in Encode",
	"日本語のテキスト",
}

func TestRoundTrip(t *testing.T) {
	tok := shared(t)
	for _, s := range roundTripCases {
		if got := tok.Decode(tok.Encode(s)); got != s {
			t.Errorf("round trip %q -> %q", s, got)
		}
		if got := tok.Decode(tok.EncodeWithSpecial(s)); got != s {
			t.Errorf("round trip with special %q -> %q", s, got)
		}
		if got := strings.Join(tok.Split(s), ""); got != s {
			t.Errorf("split %q does not concatenate back: %q", s, got)
		}
	}
}

func FuzzRoundTrip(f *testing.F) {
	for _, s := range roundTripCases {
		f.Add(s)
	}
	f.Fuzz(func(t *testing.T, s string) {
		tok := shared(t)
		ids := tok.Encode(s)
		if got := tok.Decode(ids); got != s {
			t.Fatalf("Decode(Encode(%q)) = %q", s, got)
		}
		for _, id := range ids {
			if _, special := tok.specialByID[id]; special {
				t.Fatalf("Encode produced special token %d for %q", id, s)
			}
		}
	})
}

func TestSpecialTokens(t *testing.T) {
	tok := shared(t)
	eot, ok := tok.SpecialID("<|endoftext|>")
	if !ok || eot != 512 || tok.VocabSize() != 514 {
		t.Fatalf("eot id = %d, vocab = %d", eot, tok.VocabSize())
	}
	text := "привет<|endoftext|>мир"
	if slices.Contains(tok.Encode(text), eot) {
		t.Fatal("Encode must not produce special tokens from user text")
	}
	ids := tok.EncodeWithSpecial(text)
	if !slices.Contains(ids, eot) {
		t.Fatalf("EncodeWithSpecial did not find the special token: %v", ids)
	}
	if tok.Decode(ids) != text {
		t.Fatal("special tokens must decode to their text")
	}
}

func TestSaveLoad(t *testing.T) {
	tok := shared(t)
	var buf bytes.Buffer
	if err := tok.Save(&buf); err != nil {
		t.Fatal(err)
	}
	loaded, err := Load(&buf)
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(loaded.Merges(), tok.Merges()) || loaded.VocabSize() != tok.VocabSize() || loaded.Pattern() != tok.Pattern() {
		t.Fatal("loaded tokenizer differs")
	}
	for _, s := range roundTripCases {
		if !slices.Equal(loaded.EncodeWithSpecial(s), tok.EncodeWithSpecial(s)) {
			t.Fatalf("encodings differ for %q", s)
		}
	}
	if _, err := Load(strings.NewReader("not a tokenizer")); err == nil {
		t.Fatal("Load must reject garbage")
	}
}

func TestTrainErrors(t *testing.T) {
	if _, err := Train("abc", 100, Options{}); err == nil {
		t.Fatal("vocabSize < 256 must fail")
	}
	if _, err := Train("abc", 300, Options{Pattern: `\s+(?!\S)`}); err == nil {
		t.Fatal("lookahead outside the GPT tail must be rejected by RE2")
	}
	tok := mustTrain(t, "ab", 1000, Options{}) // only one pair exists: training stops early
	if len(tok.Merges()) != 1 {
		t.Fatalf("merges = %v", tok.Merges())
	}
}

func countWords(s string) int {
	return len(strings.FieldsFunc(s, func(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }))
}

// TestTokenCostRuVsEn measures the lecture's main practical point: the same meaning costs
// very different numbers of tokens depending on what the tokenizer was trained on.
// Run with -v to see the table.
func TestTokenCostRuVsEn(t *testing.T) {
	ruText, enText := readData(t, "parallel_ru.txt"), readData(t, "parallel_en.txt")
	corpora := []struct{ name, text string }{
		{"ru+en+code", readData(t, "corpus_ru.txt") + enText + readData(t, "code_go.txt")},
		{"en+code", enText + readData(t, "code_go.txt")},
	}
	ratios := map[string]float64{}
	for _, c := range corpora {
		for _, vs := range []int{256, 512, 1024, 2048} {
			tok := mustTrain(t, c.text, vs, Options{Pattern: PatternGPT4})
			ru, en := len(tok.Encode(ruText)), len(tok.Encode(enText))
			t.Logf("%-10s vocab=%4d  ru: %4d tok %5.2f tok/word %4.2f B/tok | en: %4d tok %5.2f tok/word %4.2f B/tok | ru/en %.2f",
				c.name, vs, ru, float64(ru)/float64(countWords(ruText)), float64(len(ruText))/float64(ru),
				en, float64(en)/float64(countWords(enText)), float64(len(enText))/float64(en), float64(ru)/float64(en))
			if vs == 1024 {
				ratios[c.name] = float64(ru) / float64(en)
			}
		}
	}
	if ratios["ru+en+code"] > 1.3 || ratios["en+code"] < 3 {
		t.Fatalf("unexpected ru/en ratios: %v", ratios)
	}
}

func BenchmarkTrain(b *testing.B) {
	text := readData(b, "corpus_ru.txt") + readData(b, "parallel_en.txt") + readData(b, "code_go.txt")
	b.SetBytes(int64(len(text)))
	for b.Loop() {
		mustTrain(b, text, 1024, Options{Pattern: PatternGPT4})
	}
}

func BenchmarkEncode(b *testing.B) {
	text := readData(b, "corpus_ru.txt") + readData(b, "parallel_en.txt") + readData(b, "code_go.txt")
	tok := mustTrain(b, text, 1024, Options{Pattern: PatternGPT4})
	b.SetBytes(int64(len(text)))
	for b.Loop() {
		tok.Encode(text)
	}
}
