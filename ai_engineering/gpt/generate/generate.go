package generate

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"math"
	"strings"
	"time"

	"algorithms_and_data_structures/ai_engineering/gpt"
)

// Tokenizer is what generation needs: encode the prompt, map an id to its bytes.
// train.CharTokenizer and tokenizer.Tokenizer (BPE, L5) implement it.
type Tokenizer interface {
	Encode(text string) []int
	TokenBytes(id int) []byte
	VocabSize() int
}

// Config of one generation.
type Config struct {
	MaxTokens  int           // new tokens at most (0 = Context); longer texts slide the window
	Sampler    SamplerConfig // decoding strategy
	Stop       []string      // stop sequences: generation ends before the first occurrence
	EOS        []int         // token ids that end generation (e.g. <|endoftext|>)
	Constraint Constraint    // optional logit mask (JSON, Schema); generation ends when it is Done
	NoCache    bool          // recompute the whole sequence every step (for comparison: O(T²) work)
}

// Token is one generated token. Text is the part of the output that became
// printable with this token: "" while a UTF-8 character or a possible stop sequence
// is incomplete, several characters when a held-back tail is released.
type Token struct {
	ID   int
	Text string
}

// Stats describe a finished generation.
type Stats struct {
	PromptTokens, Tokens int
	Prefill, Decode      time.Duration // time to first token, time of all other steps
	Finish               string        // "length", "stop", "eos", "constraint", "cancelled"
}

// ErrNoAllowedToken means the constraint forbids every token of the vocabulary
// (e.g. JSON needs '{' but a character vocabulary has no such character).
var ErrNoAllowedToken = errors.New("generate: constraint allows no token of the vocabulary")

// Generate streams the continuation of prompt token by token:
//
//	for tok, err := range generate.Generate(ctx, m, tk, "Кот", cfg) {
//		if err != nil { … }
//		fmt.Print(tok.Text)
//	}
//
// The prompt is prefilled in one forward pass; every next token costs one decode
// step over the KV-cache (or a full forward with cfg.NoCache).
func Generate(ctx context.Context, m *gpt.Model, tok Tokenizer, prompt string, cfg Config) iter.Seq2[Token, error] {
	return func(yield func(Token, error) bool) {
		stopped := false
		_, err := run(ctx, m, tok, tok.Encode(prompt), cfg, func(t Token) bool {
			if !yield(t, nil) {
				stopped = true
				return false
			}
			return true
		})
		if err != nil && !stopped {
			yield(Token{ID: -1}, err)
		}
	}
}

// Text generates and returns the whole continuation.
func Text(ctx context.Context, m *gpt.Model, tok Tokenizer, prompt string, cfg Config) (string, Stats, error) {
	var sb strings.Builder
	st, err := run(ctx, m, tok, tok.Encode(prompt), cfg, func(t Token) bool {
		sb.WriteString(t.Text)
		return true
	})
	return sb.String(), st, err
}

// IDs generates from prompt ids and returns the new ids (no text handling except
// stop sequences, which need text). Used by tests and benchmarks.
func IDs(ctx context.Context, m *gpt.Model, tok Tokenizer, prompt []int, cfg Config) ([]int, Stats, error) {
	var ids []int
	st, err := run(ctx, m, tok, prompt, cfg, func(t Token) bool {
		if t.ID >= 0 { // ID −1 carries only released text at the end
			ids = append(ids, t.ID)
		}
		return true
	})
	return ids, st, err
}

func run(ctx context.Context, m *gpt.Model, tok Tokenizer, prompt []int, cfg Config, emit func(Token) bool) (Stats, error) {
	st := Stats{PromptTokens: len(prompt)}
	if len(prompt) == 0 {
		return st, fmt.Errorf("generate: empty prompt")
	}
	if len(prompt) > m.Cfg.Context {
		return st, fmt.Errorf("generate: prompt of %d tokens does not fit context %d", len(prompt), m.Cfg.Context)
	}
	maxNew := cfg.MaxTokens
	if maxNew <= 0 {
		maxNew = m.Cfg.Context
	}
	sampler := NewSampler(cfg.Sampler)
	var sess *gpt.Session
	if !cfg.NoCache {
		sess = m.NewSession(1, 0)
	}
	ids := append([]int(nil), prompt...) // everything so far (penalties look at it)
	win := 0                             // the model sees ids[win:], at positions 0, 1, …
	// next returns the logits of the token after ids; with the cache only the tokens
	// the session has not seen yet are fed.
	next := func() []float32 {
		if len(ids)-win > m.Cfg.Context {
			// Context full: slide the window — keep the last half and start over.
			// Positions change, so the cache is useless: reset and prefill again
			// (llama.cpp's "context shift" avoids this by re-rotating cached RoPE keys).
			win = len(ids) - m.Cfg.Context/2
			if sess != nil {
				sess.Reset()
			}
		}
		v := m.Cfg.Vocab
		if sess == nil {
			out := m.Forward([][]int{ids[win:]}).Data
			return out.Data()[(len(ids)-win-1)*v:]
		}
		d := sess.Forward([][]int{ids[win+sess.Len():]}).Data()
		return d[len(d)-v:]
	}

	var detok Detokenizer
	stops := stopper{stops: cfg.Stop}
	start := time.Now()
	logits := next()
	st.Prefill = time.Since(start)
	// finish flushes held-back text and reports the reason.
	finish := func(reason string) (Stats, error) {
		st.Finish = reason
		st.Decode = time.Since(start) - st.Prefill
		if reason != "stop" {
			if tail, _ := stops.push(detok.Flush()); tail != "" || stops.pending != "" {
				emit(Token{ID: -1, Text: tail + stops.flush()})
			}
		}
		return st, nil
	}
	for {
		if err := ctx.Err(); err != nil {
			return st, err
		}
		if cfg.Constraint != nil {
			logits = mask(logits, tok, cfg.Constraint)
			if logits == nil {
				return st, ErrNoAllowedToken
			}
		}
		id := sampler.Sample(logits, ids)
		ids = append(ids, id)
		st.Tokens++
		b := tok.TokenBytes(id)
		if cfg.Constraint != nil {
			cfg.Constraint.Advance(b)
		}
		text, stopped := stops.push(detok.Push(b))
		if !emit(Token{ID: id, Text: text}) {
			st.Finish = "cancelled" // the consumer stopped reading: emit nothing more
			return st, nil
		}
		switch {
		case stopped:
			return finish("stop")
		case contains(cfg.EOS, id):
			return finish("eos")
		case cfg.Constraint != nil && cfg.Constraint.Done():
			return finish("constraint")
		case st.Tokens >= maxNew:
			return finish("length")
		}
		logits = next()
	}
}

// mask returns a copy of logits with every token the constraint forbids set to −∞,
// or nil if nothing is allowed.
func mask(logits []float32, tok Tokenizer, c Constraint) []float32 {
	out := make([]float32, len(logits))
	inf := float32(math.Inf(-1))
	any := false
	for id, z := range logits {
		if c.Allowed(tok.TokenBytes(id)) {
			out[id] = z
			any = true
		} else {
			out[id] = inf
		}
	}
	if !any {
		return nil
	}
	return out
}

func contains(xs []int, x int) bool {
	for _, y := range xs {
		if x == y {
			return true
		}
	}
	return false
}
