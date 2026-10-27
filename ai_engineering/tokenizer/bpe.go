// Package tokenizer implements byte-level BPE (byte pair encoding), the tokenization
// scheme of GPT-2/3/4 and Llama 3 (lecture L5).
//
// Token ids 0..255 are raw bytes, so any string — any language, emoji, even invalid
// UTF-8 — can be encoded, and Decode(Encode(s)) == s always holds. Each learned merge
// (a, b) -> 256+i glues two existing tokens into a new one. Special tokens get the ids
// after the merges.
package tokenizer

import (
	"container/heap"
	"fmt"
	"regexp"
	"slices"
	"sort"
	"strings"
)

// Pair is two adjacent token ids.
type Pair struct{ A, B int }

// Options configure training.
type Options struct {
	// Pattern is the pre-tokenization regex (PatternGPT4, PatternGPT2, PatternNone or your own).
	Pattern string
	// SpecialTokens get ids after all merges, in the given order.
	SpecialTokens []string
}

// Tokenizer is a trained byte-level BPE tokenizer. It is safe for concurrent use.
type Tokenizer struct {
	pattern     string
	split       *splitter
	merges      []Pair       // merges[i] creates token 256+i
	ranks       map[Pair]int // pair -> id of the merged token (lower id = learned earlier)
	vocab       [][]byte     // id -> bytes, for byte and merged tokens
	special     map[string]int
	specialByID map[int]string
	specialRe   *regexp.Regexp
}

// Train learns vocabSize-256 merges on text. vocabSize counts bytes and merges;
// special tokens come on top (see VocabSize). Training stops early if no pair is left.
//
// Ties between equally frequent pairs are broken by the smaller pair (A, then B),
// so training is fully deterministic.
func Train(text string, vocabSize int, opts Options) (*Tokenizer, error) {
	if vocabSize < 256 {
		return nil, fmt.Errorf("tokenizer: vocabSize %d < 256 byte tokens", vocabSize)
	}
	split, err := newSplitter(opts.Pattern)
	if err != nil {
		return nil, err
	}

	// 1. Unique chunks with their frequencies: " the" may occur 10 000 times, but we
	//    store it once with freq 10 000 and multiply its pair counts.
	freq := map[string]int{}
	for _, c := range split.split(text) {
		freq[c]++
	}
	keys := make([]string, 0, len(freq))
	for k := range freq {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	words := make([][]int, len(keys))
	counts := make([]int, len(keys))
	for i, k := range keys {
		words[i] = bytesToIDs([]byte(k))
		counts[i] = freq[k]
	}

	// 2. Count all adjacent pairs and remember which words contain each pair.
	st := &trainState{pairs: map[Pair]int{}, where: map[Pair]map[int]struct{}{}}
	for w, ids := range words {
		st.addWord(w, ids, counts[w], +1)
	}
	st.flush()

	// 3. Repeatedly merge the most frequent pair, updating counts only for affected words.
	merges := make([]Pair, 0, vocabSize-256)
	for len(merges) < vocabSize-256 {
		best, ok := st.popBest()
		if !ok {
			break
		}
		id := 256 + len(merges)
		merges = append(merges, best)
		for w := range st.where[best] {
			if !containsPair(words[w], best) {
				continue // stale index entry: the pair was destroyed by an earlier merge
			}
			st.addWord(w, words[w], counts[w], -1)
			words[w] = mergePair(words[w], best, id)
			st.addWord(w, words[w], counts[w], +1)
		}
		delete(st.where, best)
		st.flush()
	}
	return build(opts.Pattern, merges, opts.SpecialTokens)
}

// trainState holds incremental pair statistics during training.
type trainState struct {
	pairs   map[Pair]int              // pair -> total count over the corpus
	where   map[Pair]map[int]struct{} // pair -> indices of words that (may) contain it
	queue   pairQueue                 // max-heap of (pair, count); may hold stale entries
	changed map[Pair]struct{}
}

// addWord adds (sign=+1) or removes (sign=-1) the pairs of one word, weighted by its frequency.
func (st *trainState) addWord(w int, ids []int, freq, sign int) {
	if st.changed == nil {
		st.changed = map[Pair]struct{}{}
	}
	for i := 0; i+1 < len(ids); i++ {
		p := Pair{ids[i], ids[i+1]}
		st.pairs[p] += sign * freq
		if st.pairs[p] == 0 {
			delete(st.pairs, p)
		}
		if sign > 0 {
			if st.where[p] == nil {
				st.where[p] = map[int]struct{}{}
			}
			st.where[p][w] = struct{}{}
		}
		st.changed[p] = struct{}{}
	}
}

// flush pushes fresh heap entries for every pair whose count changed (lazy deletion:
// old entries stay in the heap and are skipped when popped).
func (st *trainState) flush() {
	for p := range st.changed {
		if c := st.pairs[p]; c > 0 {
			heap.Push(&st.queue, pairCount{p, c})
		}
	}
	clear(st.changed)
}

func (st *trainState) popBest() (Pair, bool) {
	for st.queue.Len() > 0 {
		top := heap.Pop(&st.queue).(pairCount)
		if st.pairs[top.p] == top.c { // skip stale entries
			return top.p, true
		}
	}
	return Pair{}, false
}

type pairCount struct {
	p Pair
	c int
}

// pairQueue is a max-heap by count; ties go to the smaller pair.
type pairQueue []pairCount

func (q pairQueue) Len() int { return len(q) }
func (q pairQueue) Less(i, j int) bool {
	if q[i].c != q[j].c {
		return q[i].c > q[j].c
	}
	if q[i].p.A != q[j].p.A {
		return q[i].p.A < q[j].p.A
	}
	return q[i].p.B < q[j].p.B
}
func (q pairQueue) Swap(i, j int) { q[i], q[j] = q[j], q[i] }
func (q *pairQueue) Push(x any)   { *q = append(*q, x.(pairCount)) }
func (q *pairQueue) Pop() any {
	old := *q
	x := old[len(old)-1]
	*q = old[:len(old)-1]
	return x
}

// build assembles a tokenizer from a pattern, merges and special tokens.
func build(pattern string, merges []Pair, specials []string) (*Tokenizer, error) {
	split, err := newSplitter(pattern)
	if err != nil {
		return nil, err
	}
	t := &Tokenizer{
		pattern:     pattern,
		split:       split,
		merges:      slices.Clone(merges),
		ranks:       make(map[Pair]int, len(merges)),
		vocab:       make([][]byte, 256, 256+len(merges)),
		special:     map[string]int{},
		specialByID: map[int]string{},
	}
	for b := range 256 {
		t.vocab[b] = []byte{byte(b)}
	}
	for i, p := range merges {
		id := 256 + i
		if p.A >= id || p.B >= id || p.A < 0 || p.B < 0 {
			return nil, fmt.Errorf("tokenizer: merge %d (%d, %d) refers to an unknown token", i, p.A, p.B)
		}
		t.ranks[p] = id
		t.vocab = append(t.vocab, append(slices.Clone(t.vocab[p.A]), t.vocab[p.B]...))
	}
	if len(specials) > 0 {
		quoted := make([]string, len(specials))
		for i, s := range specials {
			if s == "" {
				return nil, fmt.Errorf("tokenizer: empty special token")
			}
			if _, dup := t.special[s]; dup {
				return nil, fmt.Errorf("tokenizer: duplicate special token %q", s)
			}
			id := len(t.vocab) + i
			t.special[s] = id
			t.specialByID[id] = s
			quoted[i] = regexp.QuoteMeta(s)
		}
		// Longest first, so "<|end|>" never wins over "<|endoftext|>".
		sort.Slice(quoted, func(i, j int) bool { return len(quoted[i]) > len(quoted[j]) })
		t.specialRe = regexp.MustCompile(strings.Join(quoted, "|"))
	}
	return t, nil
}

// Encode turns text into token ids. Special-token strings inside text are treated as
// ordinary text — this is the safe default for untrusted input (see EncodeWithSpecial).
func (t *Tokenizer) Encode(text string) []int {
	var ids []int
	for _, chunk := range t.split.split(text) {
		ids = append(ids, t.encodeChunk([]byte(chunk))...)
	}
	return ids
}

// EncodeWithSpecial is like Encode but maps occurrences of special-token strings to
// their ids. Use it only for text you control (chat templates), never for user input.
func (t *Tokenizer) EncodeWithSpecial(text string) []int {
	if t.specialRe == nil {
		return t.Encode(text)
	}
	var ids []int
	last := 0
	for _, m := range t.specialRe.FindAllStringIndex(text, -1) {
		ids = append(ids, t.Encode(text[last:m[0]])...)
		ids = append(ids, t.special[text[m[0]:m[1]]])
		last = m[1]
	}
	return append(ids, t.Encode(text[last:])...)
}

// encodeChunk applies merges to one chunk: start from bytes and repeatedly merge the
// adjacent pair that was learned earliest (lowest rank), exactly as in training.
func (t *Tokenizer) encodeChunk(b []byte) []int {
	ids := bytesToIDs(b)
	for len(ids) >= 2 {
		best, bestID := Pair{}, -1
		for i := 0; i+1 < len(ids); i++ {
			p := Pair{ids[i], ids[i+1]}
			if id, ok := t.ranks[p]; ok && (bestID < 0 || id < bestID) {
				best, bestID = p, id
			}
		}
		if bestID < 0 {
			break
		}
		ids = mergePair(ids, best, bestID)
	}
	return ids
}

// Decode turns ids back into text. Bytes are concatenated as is, so a prefix of a
// token sequence may end in the middle of a UTF-8 character.
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.Write(t.TokenBytes(id))
	}
	return sb.String()
}

// TokenBytes returns the bytes that token id stands for.
func (t *Tokenizer) TokenBytes(id int) []byte {
	if id >= 0 && id < len(t.vocab) {
		return t.vocab[id]
	}
	if s, ok := t.specialByID[id]; ok {
		return []byte(s)
	}
	panic(fmt.Sprintf("tokenizer: unknown token id %d", id))
}

// SpecialID returns the id of a special token.
func (t *Tokenizer) SpecialID(s string) (int, bool) {
	id, ok := t.special[s]
	return id, ok
}

// VocabSize is the number of distinct ids: 256 bytes + merges + special tokens.
func (t *Tokenizer) VocabSize() int { return len(t.vocab) + len(t.special) }

// Merges returns the learned merges in order; merges[i] produced token 256+i.
func (t *Tokenizer) Merges() []Pair { return slices.Clone(t.merges) }

// Pattern returns the pre-tokenization pattern.
func (t *Tokenizer) Pattern() string { return t.pattern }

// Split exposes pre-tokenization: the chunks Encode works on.
func (t *Tokenizer) Split(text string) []string { return t.split.split(text) }

func bytesToIDs(b []byte) []int {
	ids := make([]int, len(b))
	for i, c := range b {
		ids[i] = int(c)
	}
	return ids
}

// mergePair replaces every non-overlapping occurrence of p (scanning left to right) with id.
func mergePair(ids []int, p Pair, id int) []int {
	out := make([]int, 0, len(ids))
	for i := 0; i < len(ids); i++ {
		if i+1 < len(ids) && ids[i] == p.A && ids[i+1] == p.B {
			out = append(out, id)
			i++
			continue
		}
		out = append(out, ids[i])
	}
	return out
}

func containsPair(ids []int, p Pair) bool {
	for i := 0; i+1 < len(ids); i++ {
		if ids[i] == p.A && ids[i+1] == p.B {
			return true
		}
	}
	return false
}
