// Package train trains GoGPT from scratch (lecture L9): the data pipeline
// (text → tokens → uint16 file → random windows → batches), the training loop with
// gradient accumulation, clipping and a warmup+cosine schedule, data-parallel
// training on goroutines with a gradient all-reduce, evaluation, CSV logging and
// resumable checkpoints.
package train

import (
	"encoding/binary"
	"fmt"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"unicode/utf8"

	"algorithms_and_data_structures/ai_engineering/tokenizer"
)

// ---------- tokenizers ----------

// Tokenizer is what training and generation need from a tokenizer. Both the
// character tokenizer below and the BPE tokenizer of L5 implement it.
type Tokenizer interface {
	Encode(text string) []int
	Decode(ids []int) string
	TokenBytes(id int) []byte // the bytes of one token (may be part of a UTF-8 character)
	VocabSize() int
}

// CharTokenizer maps every distinct character (rune) of a text to an id, in sorted order.
type CharTokenizer struct {
	runes []rune
	ids   map[rune]int
}

// NewCharTokenizer builds the alphabet of text (sorted, so it is deterministic).
func NewCharTokenizer(text string) *CharTokenizer {
	set := map[rune]bool{}
	for _, r := range text {
		set[r] = true
	}
	runes := make([]rune, 0, len(set))
	for r := range set {
		runes = append(runes, r)
	}
	slices.Sort(runes)
	return charTokenizer(runes)
}

func charTokenizer(runes []rune) *CharTokenizer {
	c := &CharTokenizer{runes: runes, ids: map[rune]int{}}
	for i, r := range runes {
		c.ids[r] = i
	}
	return c
}

// Encode maps characters to ids; characters outside the alphabet are skipped.
func (c *CharTokenizer) Encode(text string) []int {
	ids := make([]int, 0, len(text))
	for _, r := range text {
		if id, ok := c.ids[r]; ok {
			ids = append(ids, id)
		}
	}
	return ids
}

func (c *CharTokenizer) Decode(ids []int) string {
	rs := make([]rune, len(ids))
	for i, id := range ids {
		rs[i] = c.runes[id]
	}
	return string(rs)
}

func (c *CharTokenizer) TokenBytes(id int) []byte { return utf8.AppendRune(nil, c.runes[id]) }
func (c *CharTokenizer) VocabSize() int           { return len(c.runes) }

// Alphabet is the vocabulary as a string (id i = i-th rune); it is stored in checkpoints.
func (c *CharTokenizer) Alphabet() string { return string(c.runes) }

// TokenizerMeta describes tok for a checkpoint's Meta.
func TokenizerMeta(tok Tokenizer) (map[string]string, error) {
	switch t := tok.(type) {
	case *CharTokenizer:
		return map[string]string{"tokenizer": "char", "alphabet": t.Alphabet()}, nil
	case *tokenizer.Tokenizer:
		var sb strings.Builder
		if err := t.Save(&sb); err != nil {
			return nil, err
		}
		return map[string]string{"tokenizer": "bpe", "bpe": sb.String()}, nil
	}
	return nil, fmt.Errorf("train: cannot store tokenizer %T", tok)
}

// TokenizerFromMeta rebuilds the tokenizer stored by TokenizerMeta.
func TokenizerFromMeta(meta map[string]string) (Tokenizer, error) {
	switch meta["tokenizer"] {
	case "char":
		return charTokenizer([]rune(meta["alphabet"])), nil
	case "bpe":
		return tokenizer.Load(strings.NewReader(meta["bpe"]))
	}
	return nil, fmt.Errorf("train: checkpoint has no tokenizer (meta %q)", meta["tokenizer"])
}

// ---------- token files ----------

// WriteTokens stores ids as little-endian uint16 — 2 bytes per token, the nanoGPT
// format. A vocabulary of up to 65 536 ids fits (GPT-2: 50 257); Llama 3 (128 256) needs uint32.
func WriteTokens(path string, ids []int) error {
	buf := make([]byte, 2*len(ids))
	for i, id := range ids {
		if id < 0 || id > 0xFFFF {
			return fmt.Errorf("train: token id %d does not fit uint16", id)
		}
		binary.LittleEndian.PutUint16(buf[2*i:], uint16(id))
	}
	return os.WriteFile(path, buf, 0o644)
}

// ReadTokens loads a file written by WriteTokens.
func ReadTokens(path string) ([]uint16, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	if len(b)%2 != 0 {
		return nil, fmt.Errorf("train: %s has an odd number of bytes", path)
	}
	out := make([]uint16, len(b)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(b[2*i:])
	}
	return out, nil
}

// ---------- batches ----------

// Dataset is a stream of tokens; training samples random windows from it.
type Dataset struct {
	Tokens []uint16
}

// NewDataset wraps ids (all must fit uint16).
func NewDataset(ids []int) *Dataset {
	t := make([]uint16, len(ids))
	for i, id := range ids {
		t[i] = uint16(id)
	}
	return &Dataset{Tokens: t}
}

// Batch draws b windows of t+1 tokens at uniform random offsets and returns inputs
// x = window[0:t] and targets y = window[1:t+1] — "the next token" at every position,
// so one window gives t training examples (one per prefix length).
func (d *Dataset) Batch(rng *rand.Rand, b, t int) (x, y [][]int) {
	if len(d.Tokens) < t+1 {
		panic(fmt.Sprintf("train: %d tokens, need at least %d for a window", len(d.Tokens), t+1))
	}
	x, y = make([][]int, b), make([][]int, b)
	for i := range b {
		off := rng.IntN(len(d.Tokens) - t) // window [off, off+t]
		x[i], y[i] = make([]int, t), make([]int, t)
		for j := range t {
			x[i][j] = int(d.Tokens[off+j])
			y[i][j] = int(d.Tokens[off+j+1])
		}
	}
	return x, y
}

// BatchRNG is the random source of micro-batch `micro` of optimizer step `step`. It
// depends only on (seed, step, micro), so a run resumed from a checkpoint sees exactly
// the batches it would have seen without the interruption.
func BatchRNG(seed uint64, step, micro int) *rand.Rand {
	return rand.New(rand.NewPCG(seed, uint64(step)<<20|uint64(micro)))
}

// EvalWindows covers the whole dataset with windows of t tokens so that every token
// after the first is predicted exactly once, with as much context as possible: windows
// start every t/2 tokens, and all but the first count only their second half
// (targets of the first half are −1, ignored by the loss). Returns inputs, targets.
func (d *Dataset) EvalWindows(t int) (x, y [][]int) {
	n := len(d.Tokens) - 1 // number of predictions
	if n < 1 {
		return nil, nil
	}
	t = min(t, n)
	stride := max(1, t/2)
	counted := 0 // predictions 0 … counted−1 are already covered
	for start := 0; counted < n; start += stride {
		start = min(start, n-t)
		xi, yi := make([]int, t), make([]int, t)
		for j := range t {
			xi[j] = int(d.Tokens[start+j])
			if start+j >= counted {
				yi[j] = int(d.Tokens[start+j+1])
			} else {
				yi[j] = -1
			}
		}
		counted = start + t
		x, y = append(x, xi), append(y, yi)
	}
	return x, y
}
