package tokenizer

import (
	"fmt"
	"regexp"
	"strings"
	"unicode/utf8"
)

// Pre-tokenization patterns. Text is first cut into chunks by the pattern, and BPE merges
// never cross chunk boundaries (so " cat" and "." can never become one token).
const (
	// PatternGPT2 is the split pattern of GPT-2.
	PatternGPT2 = `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`

	// PatternGPT4 is the split pattern of GPT-4's cl100k_base: case-insensitive contractions,
	// one optional non-letter prefix before a word, numbers in groups of at most 3 digits,
	// newlines kept together.
	PatternGPT4 = `'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`

	// PatternNone disables pre-tokenization: the whole text is one chunk.
	PatternNone = ""
)

// lookaheadTail is the only construct of the GPT patterns that Go's RE2 engine cannot
// compile: (?!\S) is a lookahead. We replace the pair of alternatives with a named group
// and emulate the lookahead in code (see splitter.split).
const (
	lookaheadTail = `\s+(?!\S)|\s+`
	wsGroup       = `(?P<ws>\s+)`
)

// splitter cuts text into pre-tokenization chunks.
type splitter struct {
	re *regexp.Regexp
	ws int // index of the "ws" submatch, or -1 if the pattern has no lookahead tail
}

func newSplitter(pattern string) (*splitter, error) {
	if pattern == PatternNone {
		return nil, nil
	}
	p := strings.Replace(pattern, lookaheadTail, wsGroup, 1)
	re, err := regexp.Compile(`^(?:` + p + `)`)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: pattern: %w", err)
	}
	return &splitter{re: re, ws: re.SubexpIndex("ws")}, nil
}

// split returns the chunks of text. Concatenating them always gives back text exactly.
//
// Emulation of `\s+(?!\S)|\s+`: when a whitespace run is matched by the fallback
// group and more text follows, the original pattern would stop one character short,
// leaving the last whitespace character to start the next chunk (so " world" keeps
// its leading space). We do the same: shorten the match by its last rune and rescan.
func (s *splitter) split(text string) []string {
	if s == nil {
		if text == "" {
			return nil
		}
		return []string{text}
	}
	var chunks []string
	for pos := 0; pos < len(text); {
		rest := text[pos:]
		m := s.re.FindStringSubmatchIndex(rest)
		end := 0
		if m != nil {
			end = m[1]
		}
		if end == 0 { // cannot happen for the GPT patterns; never loop forever
			_, end = utf8.DecodeRuneInString(rest)
		}
		if s.ws >= 0 && m != nil && m[2*s.ws] == 0 && end < len(rest) {
			if _, last := utf8.DecodeLastRuneInString(rest[:end]); last < end {
				end -= last
			}
		}
		chunks = append(chunks, rest[:end])
		pos += end
	}
	return chunks
}
