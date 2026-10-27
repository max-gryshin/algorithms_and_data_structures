package tokenizer

import (
	"bufio"
	"fmt"
	"io"
	"strconv"
	"strings"
)

// File format (text, line based):
//
//	gobpe v1
//	pattern "<Go-quoted pattern>"
//	merges <n>
//	<a> <b>          (n lines; line i creates token 256+i)
//	special <m>
//	"<Go-quoted special token>"   (m lines, ids follow the merges)
const magic = "gobpe v1"

// Save writes the tokenizer to w.
func (t *Tokenizer) Save(w io.Writer) error {
	bw := bufio.NewWriter(w)
	fmt.Fprintln(bw, magic)
	fmt.Fprintf(bw, "pattern %s\n", strconv.Quote(t.pattern))
	fmt.Fprintf(bw, "merges %d\n", len(t.merges))
	for _, p := range t.merges {
		fmt.Fprintf(bw, "%d %d\n", p.A, p.B)
	}
	fmt.Fprintf(bw, "special %d\n", len(t.special))
	for id := len(t.vocab); id < t.VocabSize(); id++ {
		fmt.Fprintln(bw, strconv.Quote(t.specialByID[id]))
	}
	return bw.Flush()
}

// Load reads a tokenizer written by Save.
func Load(r io.Reader) (*Tokenizer, error) {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 64*1024), 1<<20)
	line := func() (string, error) {
		if !sc.Scan() {
			if err := sc.Err(); err != nil {
				return "", err
			}
			return "", io.ErrUnexpectedEOF
		}
		return sc.Text(), nil
	}
	fail := func(err error) (*Tokenizer, error) { return nil, fmt.Errorf("tokenizer: load: %w", err) }

	if l, err := line(); err != nil || l != magic {
		return fail(fmt.Errorf("bad header %q (%v)", l, err))
	}
	l, err := line()
	if err != nil {
		return fail(err)
	}
	pattern, err := strconv.Unquote(strings.TrimPrefix(l, "pattern "))
	if err != nil {
		return fail(fmt.Errorf("pattern line %q: %w", l, err))
	}

	var n int
	if l, err = line(); err != nil {
		return fail(err)
	}
	if _, err := fmt.Sscanf(l, "merges %d", &n); err != nil {
		return fail(fmt.Errorf("merges line %q: %w", l, err))
	}
	merges := make([]Pair, n)
	for i := range merges {
		if l, err = line(); err != nil {
			return fail(err)
		}
		if _, err := fmt.Sscanf(l, "%d %d", &merges[i].A, &merges[i].B); err != nil {
			return fail(fmt.Errorf("merge %d %q: %w", i, l, err))
		}
	}

	if l, err = line(); err != nil {
		return fail(err)
	}
	if _, err := fmt.Sscanf(l, "special %d", &n); err != nil {
		return fail(fmt.Errorf("special line %q: %w", l, err))
	}
	specials := make([]string, n)
	for i := range specials {
		if l, err = line(); err != nil {
			return fail(err)
		}
		if specials[i], err = strconv.Unquote(l); err != nil {
			return fail(fmt.Errorf("special %d %q: %w", i, l, err))
		}
	}
	return build(pattern, merges, specials)
}
