package generate

import (
	"strings"
	"unicode/utf8"
)

// Detokenizer turns a stream of token bytes into a stream of complete UTF-8 text.
//
// A byte-level BPE token may end in the middle of a character: «ё» is two bytes
// (0xD1 0x91) and can be split between two tokens (L5). Printing every token's bytes
// as they come would show a broken character � for a moment — or forever in a
// terminal that already rendered it. The detokenizer holds back the incomplete tail
// (at most 3 bytes) until the character is complete.
type Detokenizer struct {
	buf []byte
}

// Push appends a token's bytes and returns the text that is complete now.
func (d *Detokenizer) Push(b []byte) string {
	d.buf = append(d.buf, b...)
	cut := len(d.buf) - incompleteTail(d.buf)
	out := string(d.buf[:cut])
	d.buf = append(d.buf[:0], d.buf[cut:]...)
	return out
}

// Flush returns whatever is held back (an unfinished character becomes U+FFFD).
func (d *Detokenizer) Flush() string {
	out := strings.ToValidUTF8(string(d.buf), "�")
	d.buf = d.buf[:0]
	return out
}

// incompleteTail is the length of the suffix of b that is the beginning of a
// multi-byte character still waiting for its continuation bytes.
func incompleteTail(b []byte) int {
	// Find the start of the last character: walk back over continuation bytes 10xxxxxx.
	for back := 1; back <= min(utf8.UTFMax, len(b)); back++ {
		c := b[len(b)-back]
		if c&0xC0 == 0x80 {
			continue // continuation byte, keep walking
		}
		need := 0 // total length announced by the lead byte
		switch {
		case c&0xE0 == 0xC0:
			need = 2
		case c&0xF0 == 0xE0:
			need = 3
		case c&0xF8 == 0xF0:
			need = 4
		}
		if need > back { // lead byte announces more bytes than we have: hold them
			return back
		}
		return 0
	}
	return 0 // only continuation bytes (invalid): pass them through
}

// stopper implements stop sequences on a text stream. Text that could still turn
// into a stop sequence is held back: with stop "\n\nUser:" the text "\n\nUs" must not
// be shown until the next tokens decide.
type stopper struct {
	stops   []string
	pending string
}

// push adds text; it returns the text safe to emit and whether a stop sequence was
// found (the stop sequence itself and everything after it are dropped).
func (s *stopper) push(text string) (emit string, stopped bool) {
	s.pending += text
	if len(s.stops) == 0 {
		out := s.pending
		s.pending = ""
		return out, false
	}
	first := -1
	for _, st := range s.stops {
		if i := strings.Index(s.pending, st); i >= 0 && (first < 0 || i < first) {
			first = i
		}
	}
	if first >= 0 {
		out := s.pending[:first]
		s.pending = ""
		return out, true
	}
	// Hold back the longest suffix that is a proper prefix of some stop sequence.
	hold := 0
	for _, st := range s.stops {
		for n := min(len(st)-1, len(s.pending)); n > hold; n-- {
			if strings.HasSuffix(s.pending, st[:n]) {
				hold = n
				break
			}
		}
	}
	out := s.pending[:len(s.pending)-hold]
	s.pending = s.pending[len(s.pending)-hold:]
	return out, false
}

func (s *stopper) flush() string {
	out := s.pending
	s.pending = ""
	return out
}
