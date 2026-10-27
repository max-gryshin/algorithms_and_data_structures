package tokenizer

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"sync"
)

// GPT2 is the tokenizer of OpenAI GPT-2 (and of every model that ships the files
// vocab.json + merges.txt, e.g. GPT-2 family, RoBERTa, early Llama-likes in BPE form),
// loaded from the Hugging Face files (lecture L12).
//
// Differences from our Tokenizer (L5):
//
//   - ids come from vocab.json, not from "bytes first, merges after". Byte tokens are
//     NOT ids 0..255 in byte order: "!" (byte 33) has id 0.
//   - merges.txt stores merges as pairs of strings, one per line, in the order they
//     were learned (line i has rank i). The strings are byte-to-unicode encoded (see
//     bytesToUnicode): every byte is shown as a printable character, so the space
//     0x20 is "Ġ" (U+0120) and "\n" is "Ċ" (U+010A).
//
// It implements generate.Tokenizer. Safe for concurrent use.
type GPT2 struct {
	encoder map[string]int    // unicode-encoded token string -> id
	decoder [][]byte          // id -> raw bytes
	ranks   map[[2]string]int // merge pair -> rank (lower = learned earlier)
	split   *splitter
	special map[string]int
	byteEnc [256]string // byte -> its printable rune as a string

	mu    sync.Mutex
	cache map[string][]int // chunk -> ids (chunks repeat a lot: " the", ",")
}

// EndOfText is GPT-2's only special token (id 50256), used as BOS and EOS.
const EndOfText = "<|endoftext|>"

// patternGPT2Unicode is PatternGPT2 with Python's Unicode-aware \s. In Go's RE2 \s is
// ASCII-only ([\t\n\f\r ]), while the Python `regex` module used by OpenAI also treats
// \v, U+0085, U+00A0, U+2000…U+200A, U+3000 etc. as whitespace. With plain \s a
// no-break space would be glued to the next word differently than in the reference.
// The lookahead tail `\s+(?!\S)|\s+` is already replaced by the ws group (see splitter).
var patternGPT2Unicode = func() string {
	const ws = `[\t\n\v\f\r \x{1c}-\x{1f}\x{85}\p{Z}]`
	p := `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^WS\p{L}\p{N}]+|(?P<ws>WS+)`
	p = strings.Replace(p, "[^WS", `[^`+ws[1:len(ws)-1], 1)
	return strings.Replace(p, "WS+", ws+"+", 1)
}()

// bytesToUnicode is GPT-2's reversible byte → printable rune map: the 188 printable
// bytes ('!'..'~', '¡'..'¬', '®'..'ÿ') map to themselves, the other 68 (control
// characters, space, 0x7F–0xA0, 0xAD) map to U+0100, U+0101, … in byte order. So the
// strings in vocab.json and merges.txt never contain spaces or control characters.
func bytesToUnicode() [256]rune {
	var m [256]rune
	printable := func(b int) bool {
		return (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF)
	}
	n := 0
	for b := range 256 {
		if printable(b) {
			m[b] = rune(b)
		} else {
			m[b] = rune(256 + n)
			n++
		}
	}
	return m
}

// LoadGPT2 reads vocab.json and merges.txt.
func LoadGPT2(vocab, merges io.Reader) (*GPT2, error) {
	var enc map[string]int
	if err := json.NewDecoder(vocab).Decode(&enc); err != nil {
		return nil, fmt.Errorf("tokenizer: vocab.json: %w", err)
	}
	t := &GPT2{encoder: enc, ranks: map[[2]string]int{}, special: map[string]int{}, cache: map[string][]int{}}
	b2u := bytesToUnicode()
	u2b := map[rune]byte{}
	for b, r := range b2u {
		u2b[r] = byte(b)
		t.byteEnc[b] = string(r)
	}
	maxID := -1
	for _, id := range enc {
		if id < 0 {
			return nil, fmt.Errorf("tokenizer: vocab.json: negative id %d", id)
		}
		maxID = max(maxID, id)
	}
	t.decoder = make([][]byte, maxID+1)
	for s, id := range enc {
		if strings.HasPrefix(s, "<|") && strings.HasSuffix(s, "|>") {
			t.special[s] = id
			t.decoder[id] = []byte(s)
			continue
		}
		raw := make([]byte, 0, len(s))
		for _, r := range s {
			b, ok := u2b[r]
			if !ok {
				return nil, fmt.Errorf("tokenizer: vocab.json: token %q has rune %U outside the byte map", s, r)
			}
			raw = append(raw, b)
		}
		t.decoder[id] = raw
	}
	for b := range 256 {
		if _, ok := enc[t.byteEnc[b]]; !ok {
			return nil, fmt.Errorf("tokenizer: vocab.json misses the byte token for 0x%02x", b)
		}
	}

	sc := bufio.NewScanner(merges)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	rank := 0
	for line := 0; sc.Scan(); line++ {
		l := sc.Text()
		if strings.HasPrefix(l, "#version") || strings.TrimSpace(l) == "" {
			continue
		}
		a, b, ok := strings.Cut(l, " ")
		if !ok || strings.Contains(b, " ") {
			return nil, fmt.Errorf("tokenizer: merges.txt line %d: %q is not a pair", line+1, l)
		}
		if _, ok := enc[a+b]; !ok {
			return nil, fmt.Errorf("tokenizer: merges.txt line %d: %q has no id in vocab.json", line+1, a+b)
		}
		t.ranks[[2]string{a, b}] = rank
		rank++
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("tokenizer: merges.txt: %w", err)
	}
	re := regexp.MustCompile(`^(?:` + patternGPT2Unicode + `)`)
	t.split = &splitter{re: re, ws: re.SubexpIndex("ws")}
	return t, nil
}

// LoadGPT2Dir loads dir/vocab.json and dir/merges.txt.
func LoadGPT2Dir(dir string) (*GPT2, error) {
	v, err := os.Open(filepath.Join(dir, "vocab.json"))
	if err != nil {
		return nil, err
	}
	defer v.Close()
	m, err := os.Open(filepath.Join(dir, "merges.txt"))
	if err != nil {
		return nil, err
	}
	defer m.Close()
	return LoadGPT2(v, m)
}

// Encode splits text with the GPT-2 pattern and applies the merges to every chunk.
// Special-token strings are treated as plain text (see EncodeWithSpecial).
func (t *GPT2) Encode(text string) []int {
	var ids []int
	for _, chunk := range t.split.split(text) {
		ids = append(ids, t.encodeChunk(chunk)...)
	}
	return ids
}

// EncodeWithSpecial maps occurrences of special tokens (<|endoftext|>) to their ids.
func (t *GPT2) EncodeWithSpecial(text string) []int {
	var ids []int
	for text != "" {
		at, tok := len(text), ""
		for s := range t.special {
			if i := strings.Index(text, s); i >= 0 && (i < at || i == at && len(s) > len(tok)) {
				at, tok = i, s
			}
		}
		ids = append(ids, t.Encode(text[:at])...)
		if tok == "" {
			break
		}
		ids = append(ids, t.special[tok])
		text = text[at+len(tok):]
	}
	return ids
}

// encodeChunk is the BPE of the reference implementation: start from one symbol per
// byte and repeatedly merge every occurrence of the adjacent pair with the lowest rank.
func (t *GPT2) encodeChunk(chunk string) []int {
	t.mu.Lock()
	if ids, ok := t.cache[chunk]; ok {
		t.mu.Unlock()
		return ids
	}
	t.mu.Unlock()
	syms := make([]string, len(chunk))
	for i := range len(chunk) {
		syms[i] = t.byteEnc[chunk[i]]
	}
	for len(syms) > 1 {
		best, bestRank := -1, -1
		for i := 0; i+1 < len(syms); i++ {
			if r, ok := t.ranks[[2]string{syms[i], syms[i+1]}]; ok && (bestRank < 0 || r < bestRank) {
				best, bestRank = i, r
			}
		}
		if best < 0 {
			break
		}
		a, b := syms[best], syms[best+1]
		out := syms[:0:0]
		for i := 0; i < len(syms); i++ {
			if i+1 < len(syms) && syms[i] == a && syms[i+1] == b {
				out = append(out, a+b)
				i++
				continue
			}
			out = append(out, syms[i])
		}
		syms = out
	}
	ids := make([]int, len(syms))
	for i, s := range syms {
		ids[i] = t.encoder[s] // every merge result and every byte is in the vocab (checked at load)
	}
	t.mu.Lock()
	if len(t.cache) < 1<<16 {
		t.cache[chunk] = ids
	}
	t.mu.Unlock()
	return ids
}

// Decode concatenates the bytes of ids.
func (t *GPT2) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.Write(t.TokenBytes(id))
	}
	return sb.String()
}

// TokenBytes returns the raw bytes of a token (a special token returns its text).
func (t *GPT2) TokenBytes(id int) []byte {
	if id < 0 || id >= len(t.decoder) || t.decoder[id] == nil {
		panic(fmt.Sprintf("tokenizer: unknown GPT-2 token id %d", id))
	}
	return t.decoder[id]
}

// VocabSize is the number of ids (max id + 1): 50257 for GPT-2.
func (t *GPT2) VocabSize() int { return len(t.decoder) }

// SpecialID returns the id of a special token such as EndOfText.
func (t *GPT2) SpecialID(s string) (int, bool) {
	id, ok := t.special[s]
	return id, ok
}

// TokenString is the vocab.json spelling of a token ("Ġthe" for " the"), handy for
// printing tokenizations.
func (t *GPT2) TokenString(id int) string {
	var sb strings.Builder
	for _, b := range t.TokenBytes(id) {
		sb.WriteString(t.byteEnc[b])
	}
	return sb.String()
}

// Split exposes pre-tokenization.
func (t *GPT2) Split(text string) []string { return t.split.split(text) }

// WriteGPT2Files writes vocab.json and merges.txt for the given merges (pairs of raw
// byte strings, in rank order) — the inverse of LoadGPT2. Ids: the 256 byte tokens
// in byte-to-unicode order, then one id per merge, then the specials. Tests use it
// to build small synthetic tokenizer files without the network.
func WriteGPT2Files(vocab, merges io.Writer, pairs [][2]string, specials []string) error {
	b2u := bytesToUnicode()
	enc := func(s string) string {
		var sb strings.Builder
		for i := range len(s) {
			sb.WriteRune(b2u[s[i]])
		}
		return sb.String()
	}
	v := map[string]int{}
	order := make([]int, 256)
	for i := range order {
		order[i] = i
	}
	// GPT-2's vocab.json lists byte tokens sorted by their rune, so '!' gets id 0.
	sort.Slice(order, func(i, j int) bool { return b2u[order[i]] < b2u[order[j]] })
	for _, b := range order {
		v[string(b2u[b])] = len(v)
	}
	if _, err := fmt.Fprintln(merges, "#version: 0.2"); err != nil {
		return err
	}
	for _, p := range pairs {
		a, b := enc(p[0]), enc(p[1])
		if _, ok := v[a+b]; !ok {
			v[a+b] = len(v)
		}
		if _, err := fmt.Fprintf(merges, "%s %s\n", a, b); err != nil {
			return err
		}
	}
	for _, s := range specials {
		v[s] = len(v)
	}
	return json.NewEncoder(vocab).Encode(v)
}
