package gpt

import (
	"fmt"

	"algorithms_and_data_structures/ai_engineering/nn"
	"algorithms_and_data_structures/ai_engineering/tensor"
)

// Session is an inference session with a KV-cache (lecture L10): B sequences that
// grow together, one nn.KVCache per layer. Prefill feeds the whole prompt in one
// Forward call; every decode step then feeds only the newest token, and attention
// reads the keys and values of all earlier tokens from the cache instead of
// recomputing them.
//
// A Session is not safe for concurrent use; any number of sessions may share a Model.
type Session struct {
	m      *Model
	caches []*nn.KVCache
}

// NewSession allocates caches for batch sequences of up to capacity tokens
// (capacity ≤ 0 means Cfg.Context). Memory: 2·L·B·h_kv·d_head·capacity·4 bytes.
func (m *Model) NewSession(batch, capacity int) *Session {
	if capacity <= 0 || capacity > m.Cfg.Context {
		capacity = m.Cfg.Context
	}
	s := &Session{m: m}
	for range m.Blocks {
		s.caches = append(s.caches, nn.NewKVCache(batch, m.Cfg.KVHeads, m.Cfg.HeadDim(), capacity))
	}
	return s
}

// Len is the number of tokens already in the cache (= position of the next token).
func (s *Session) Len() int { return s.caches[0].Len }

// Cap is the capacity in tokens.
func (s *Session) Cap() int { return s.caches[0].Cap }

// Bytes is the memory held by the filled part of the cache.
func (s *Session) Bytes() int {
	n := 0
	for _, c := range s.caches {
		n += c.Bytes()
	}
	return n
}

// Reset empties the cache so the session can serve a new prompt without reallocating.
func (s *Session) Reset() { s.Truncate(0) }

// Truncate keeps the first n tokens and forgets the rest (rollback of rejected
// speculative tokens, or editing the end of a conversation).
func (s *Session) Truncate(n int) {
	for _, c := range s.caches {
		c.Truncate(n)
	}
}

// Forward feeds new tokens ids (B, Tn) — a prompt (prefill) or one token per
// sequence (decode) — and returns their logits (B, Tn, V). The logits equal those of
// Model.Forward on the whole sequence, bit for bit (TestSessionMatchesForward).
func (s *Session) Forward(ids [][]int) *tensor.Tensor {
	if len(ids) != s.caches[0].B {
		panic(fmt.Sprintf("gpt: session for %d sequences got %d", s.caches[0].B, len(ids)))
	}
	pos := s.Len()
	if pos+len(ids[0]) > s.Cap() {
		panic(fmt.Sprintf("gpt: session full: %d + %d tokens > capacity %d", pos, len(ids[0]), s.Cap()))
	}
	x := s.m.embed(ids, pos)
	for i, b := range s.m.Blocks {
		x = b.ForwardCached(x, s.caches[i])
	}
	return s.m.head(x).Data
}

// ForwardCached is s.Forward(ids): the KV-cache counterpart of ForwardAt.
func (m *Model) ForwardCached(ids [][]int, s *Session) *tensor.Tensor {
	if s.m != m {
		panic("gpt: session belongs to another model")
	}
	return s.Forward(ids)
}
