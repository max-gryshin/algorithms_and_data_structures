package tokenizer

import (
	"bytes"
	"slices"
	"testing"
)

func TestGPT2SyntheticFiles(t *testing.T) {
	var v, m bytes.Buffer
	pairs := [][2]string{{" ", "t"}, {"h", "e"}, {" t", "he"}, {"c", "a"}, {"ca", "t"}}
	if err := WriteGPT2Files(&v, &m, pairs, []string{EndOfText}); err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(m.Bytes(), []byte("Ġ t")) {
		t.Fatalf("space must be written as Ġ:\n%s", m.String())
	}
	tk, err := LoadGPT2(&v, &m)
	if err != nil {
		t.Fatal(err)
	}
	if tk.VocabSize() != 256+len(pairs)+1 {
		t.Fatalf("vocab %d", tk.VocabSize())
	}
	if got := string(tk.TokenBytes(0)); got != "!" { // byte tokens are in byte-to-unicode order
		t.Fatalf("id 0 = %q", got)
	}
	text := "the cat, the cat\n"
	ids := tk.Encode(text)
	if tk.Decode(ids) != text {
		t.Fatalf("round trip %q", tk.Decode(ids))
	}
	the := tk.encoder["Ġthe"]
	cat := tk.encoder["cat"]
	if !slices.Contains(ids, the) || !slices.Contains(ids, cat) || tk.TokenString(the) != "Ġthe" {
		t.Fatalf("ids %v", ids)
	}
	eot, _ := tk.SpecialID(EndOfText)
	if got := tk.EncodeWithSpecial("cat" + EndOfText); !slices.Equal(got, []int{cat, eot}) {
		t.Fatalf("special %v", got)
	}
	// Unicode whitespace: U+00A0 is \s in Python, so it is a chunk of its own, as in the reference.
	if got := tk.Split("a\u00a0b"); !slices.Equal(got, []string{"a", "\u00a0", "b"}) {
		t.Fatalf("split %q", got)
	}
}
