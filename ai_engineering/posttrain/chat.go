// Package posttrain is lecture L13: post-training of GoGPT. Chat templates with
// special tokens, SFT examples with a loss mask on the prompt, LoRA on a whole model,
// and the preference / RL losses: Bradley–Terry (reward model), DPO, the PPO clipped
// surrogate and GRPO group-relative advantages.
package posttrain

import (
	"math/rand/v2"
	"strings"

	"algorithms_and_data_structures/ai_engineering/autograd"
	"algorithms_and_data_structures/ai_engineering/gpt"
	"algorithms_and_data_structures/ai_engineering/gpt/train"
)

// Special tokens of our chat template. Each is ONE token id, appended after the
// character alphabet: the model cannot confuse them with text.
const (
	TokSystem    = "<|system|>"
	TokUser      = "<|user|>"
	TokAssistant = "<|assistant|>"
	TokEnd       = "<|end|>"
)

var specials = []string{TokSystem, TokUser, TokAssistant, TokEnd}

// ChatTokenizer is a character tokenizer plus the special tokens.
type ChatTokenizer struct {
	Chars *train.CharTokenizer
}

// NewChatTokenizer builds the alphabet of text plus the four special tokens.
func NewChatTokenizer(text string) *ChatTokenizer {
	return &ChatTokenizer{Chars: train.NewCharTokenizer(text)}
}

// Special returns the id of a special token.
func (c *ChatTokenizer) Special(s string) int {
	for i, sp := range specials {
		if sp == s {
			return c.Chars.VocabSize() + i
		}
	}
	panic("posttrain: unknown special token " + s)
}

// EncodeText encodes plain text: a literal "<|end|>" typed by a user stays characters
// (the defence against special-token injection).
func (c *ChatTokenizer) EncodeText(text string) []int { return c.Chars.Encode(text) }

// Encode parses special tokens in text (for templates written by our code only).
func (c *ChatTokenizer) Encode(text string) []int {
	var ids []int
	for text != "" {
		best, at := -1, len(text)
		for i, sp := range specials {
			if j := strings.Index(text, sp); j >= 0 && j < at {
				best, at = i, j
			}
		}
		ids = append(ids, c.Chars.Encode(text[:at])...)
		if best < 0 {
			break
		}
		ids = append(ids, c.Chars.VocabSize()+best)
		text = text[at+len(specials[best]):]
	}
	return ids
}

func (c *ChatTokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		sb.Write(c.TokenBytes(id))
	}
	return sb.String()
}

func (c *ChatTokenizer) TokenBytes(id int) []byte {
	if n := c.Chars.VocabSize(); id >= n {
		return []byte(specials[id-n])
	}
	return c.Chars.TokenBytes(id)
}

func (c *ChatTokenizer) VocabSize() int { return c.Chars.VocabSize() + len(specials) }

// Message is one turn of a conversation.
type Message struct {
	Role    string // "system", "user", "assistant"
	Content string
}

func roleToken(role string) string {
	switch role {
	case "system":
		return TokSystem
	case "user":
		return TokUser
	case "assistant":
		return TokAssistant
	}
	panic("posttrain: unknown role " + role)
}

// Format renders msgs with the template <|role|>content<|end|>…; with
// generationPrompt it appends "<|assistant|>" — the cue for the model to answer.
func Format(msgs []Message, generationPrompt bool) string {
	var sb strings.Builder
	for _, m := range msgs {
		sb.WriteString(roleToken(m.Role) + m.Content + TokEnd)
	}
	if generationPrompt {
		sb.WriteString(TokAssistant)
	}
	return sb.String()
}

// Example is a tokenized conversation; Train[i] says whether token i is a target
// of the loss (assistant content and its <|end|>: the model must learn to stop).
type Example struct {
	IDs   []int
	Train []bool
}

// Tokenize renders msgs and marks the tokens the model is trained on.
func Tokenize(tok *ChatTokenizer, msgs []Message) Example {
	var ex Example
	add := func(ids []int, train bool) {
		for _, id := range ids {
			ex.IDs = append(ex.IDs, id)
			ex.Train = append(ex.Train, train)
		}
	}
	for _, m := range msgs {
		a := m.Role == "assistant"
		add([]int{tok.Special(roleToken(m.Role))}, false)
		add(tok.EncodeText(m.Content), a)
		add([]int{tok.Special(TokEnd)}, a)
	}
	return ex
}

// XY shifts the example into inputs and targets; with maskPrompt the targets of
// non-trained tokens are −1 (gpt.Model.Loss ignores them).
func (e Example) XY(maskPrompt bool) (x, y []int) {
	n := len(e.IDs) - 1
	x, y = append([]int(nil), e.IDs[:n]...), make([]int, n)
	for j := range n {
		y[j] = e.IDs[j+1]
		if maskPrompt && !e.Train[j+1] {
			y[j] = -1
		}
	}
	return x, y
}

// Batch samples b examples and pads them to the longest one: inputs with pad,
// targets with −1.
func Batch(rng *rand.Rand, exs []Example, b, pad int, maskPrompt bool) (x, y [][]int) {
	t := 0
	for range b {
		e := exs[rng.IntN(len(exs))]
		xi, yi := e.XY(maskPrompt)
		x, y = append(x, xi), append(y, yi)
		t = max(t, len(xi))
	}
	for i := range x {
		for len(x[i]) < t {
			x[i], y[i] = append(x[i], pad), append(y[i], -1)
		}
	}
	return x, y
}

// ApplyLoRA attaches rank-r adapters to the attention projections and MLP matrices of
// every block, freezes all base weights (RequiresGrad = false: no gradient is even
// computed for them) and returns the adapter parameters — pass them as
// train.Config.Params.
func ApplyLoRA(m *gpt.Model, rng *rand.Rand, rank int, alpha float32) []*autograd.Var {
	for _, p := range m.Parameters() {
		p.RequiresGrad = false
	}
	var ps []*autograd.Var
	for _, b := range m.Blocks {
		for _, l := range []*nnLinear{b.Attn.Q, b.Attn.K, b.Attn.V, b.Attn.O} {
			ps = append(ps, l.AddLoRA(rng, rank, alpha).Parameters()...)
		}
	}
	return ps
}

// MergeLoRA folds every adapter into its base matrix (zero inference overhead).
func MergeLoRA(m *gpt.Model) {
	for _, b := range m.Blocks {
		for _, l := range []*nnLinear{b.Attn.Q, b.Attn.K, b.Attn.V, b.Attn.O} {
			l.Merge()
		}
	}
}
