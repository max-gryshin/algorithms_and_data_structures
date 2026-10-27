package llm

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"sync"
	"time"
)

// TokenBucket is a weighted token bucket with continuous refill.
// Unlike the channel-based bucket in problems/concurrency/rate_limiter (one token per request),
// LLM limits are counted in requests AND in tokens per minute, so one call may take 3 000 units at once.
//
// State: tokens(t) = min(Capacity, tokens(t₀) + Rate·(t − t₀)).
type TokenBucket struct {
	Capacity float64 // burst size
	Rate     float64 // units added per second

	mu     sync.Mutex
	tokens float64
	last   time.Time
	now    func() time.Time
}

// NewTokenBucket returns a full bucket. perMinute is the sustained limit (e.g. RPM or TPM).
func NewTokenBucket(capacity, perMinute float64) *TokenBucket {
	return &TokenBucket{Capacity: capacity, Rate: perMinute / 60, tokens: capacity, now: time.Now}
}

func (b *TokenBucket) refill(t time.Time) {
	if !b.last.IsZero() {
		b.tokens = min(b.Capacity, b.tokens+b.Rate*t.Sub(b.last).Seconds())
	}
	b.last = t
}

// reserve takes n units if available, otherwise reports how long to wait for them.
func (b *TokenBucket) reserve(n float64) (wait time.Duration, err error) {
	if n > b.Capacity {
		return 0, fmt.Errorf("llm: request needs %.0f units, bucket capacity is %.0f", n, b.Capacity)
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.refill(b.now())
	if b.tokens >= n {
		b.tokens -= n
		return 0, nil
	}
	return time.Duration((n - b.tokens) / b.Rate * float64(time.Second)), nil
}

// TryTake takes n units without waiting.
func (b *TokenBucket) TryTake(n float64) bool {
	wait, err := b.reserve(n)
	return err == nil && wait == 0
}

// Wait blocks until n units are taken or ctx is done.
func (b *TokenBucket) Wait(ctx context.Context, n float64) error {
	for {
		wait, err := b.reserve(n)
		if err != nil || wait == 0 {
			return err
		}
		t := time.NewTimer(wait)
		select {
		case <-t.C:
		case <-ctx.Done():
			t.Stop()
			return ctx.Err()
		}
	}
}

// EstimateTokens is a crude pre-flight estimate (≈ 4 bytes per token for English; Cyrillic is
// 2 bytes per letter in UTF-8, which roughly matches its higher token cost). Use the provider's
// token-counting endpoint when precision matters.
func EstimateTokens(req Request) float64 {
	n := len(req.System)
	for _, m := range req.Messages {
		n += len(m.Content)
		for _, c := range m.ToolCalls {
			n += len(c.Name) + len(c.Input)
		}
		for _, r := range m.ToolResults {
			n += len(r.Content)
		}
	}
	for _, t := range req.Tools {
		schema, _ := json.Marshal(t.InputSchema)
		n += len(t.Name) + len(t.Description) + len(schema)
	}
	return float64(n)/4 + float64(req.MaxTokens)
}

// WithRateLimit makes every call take one unit from requests and EstimateTokens(req) from tokens.
// Either bucket may be nil.
func WithRateLimit(c Client, requests, tokens *TokenBucket) Client {
	return &rateLimited{next: c, requests: requests, tokens: tokens}
}

type rateLimited struct {
	next             Client
	requests, tokens *TokenBucket
}

func (r *rateLimited) acquire(ctx context.Context, req Request) error {
	if r.requests != nil {
		if err := r.requests.Wait(ctx, 1); err != nil {
			return err
		}
	}
	if r.tokens != nil {
		return r.tokens.Wait(ctx, EstimateTokens(req))
	}
	return nil
}

func (r *rateLimited) Complete(ctx context.Context, req Request) (*Response, error) {
	if err := r.acquire(ctx, req); err != nil {
		return nil, err
	}
	return r.next.Complete(ctx, req)
}

func (r *rateLimited) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		if err := r.acquire(ctx, req); err != nil {
			yield(Event{}, err)
			return
		}
		for ev, err := range r.next.Stream(ctx, req) {
			if !yield(ev, err) || err != nil {
				return
			}
		}
	}
}
