package llm

import (
	"context"
	"errors"
	"iter"
	"sync"
	"time"
)

// ErrCircuitOpen is returned without calling the provider while the breaker is open.
var ErrCircuitOpen = errors.New("llm: circuit breaker is open")

type BreakerState int

const (
	Closed   BreakerState = iota // calls pass; consecutive failures are counted
	Open                         // calls fail fast until Cooldown passes
	HalfOpen                     // one probe call is let through to test recovery
)

// Breaker is a consecutive-failures circuit breaker. It protects a struggling provider
// (and our own latency) from a retry storm: after Threshold failures in a row we stop
// calling for Cooldown, then let a single probe through.
type Breaker struct {
	Threshold int
	Cooldown  time.Duration

	mu       sync.Mutex
	state    BreakerState
	failures int
	openedAt time.Time
	probing  bool
	now      func() time.Time
}

func NewBreaker(threshold int, cooldown time.Duration) *Breaker {
	return &Breaker{Threshold: threshold, Cooldown: cooldown, now: time.Now}
}

// State returns the current state (moving Open → HalfOpen if the cooldown has passed).
func (b *Breaker) State() BreakerState {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.advance()
	return b.state
}

func (b *Breaker) advance() {
	if b.state == Open && b.now().Sub(b.openedAt) >= b.Cooldown {
		b.state, b.probing = HalfOpen, false
	}
}

func (b *Breaker) allow() error {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.advance()
	switch b.state {
	case Open:
		return ErrCircuitOpen
	case HalfOpen:
		if b.probing {
			return ErrCircuitOpen
		}
		b.probing = true
	}
	return nil
}

// record counts only failures that indicate provider health problems (retryable ones):
// a 400 caused by our own bad request says nothing about the provider.
func (b *Breaker) record(err error) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if err == nil || !IsRetryable(err) {
		b.state, b.failures, b.probing = Closed, 0, false
		return
	}
	b.failures++
	if b.state == HalfOpen || b.failures >= b.Threshold {
		b.state, b.openedAt, b.probing = Open, b.now(), false
	}
}

// WithBreaker wraps c with b. Put it outside WithRetry so that one exhausted retry
// sequence counts as one failure.
func WithBreaker(c Client, b *Breaker) Client { return &breaking{next: c, b: b} }

type breaking struct {
	next Client
	b    *Breaker
}

func (w *breaking) Complete(ctx context.Context, req Request) (*Response, error) {
	if err := w.b.allow(); err != nil {
		return nil, err
	}
	resp, err := w.next.Complete(ctx, req)
	w.b.record(err)
	return resp, err
}

func (w *breaking) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		if err := w.b.allow(); err != nil {
			yield(Event{}, err)
			return
		}
		var failure error
		defer func() { w.b.record(failure) }()
		for ev, err := range w.next.Stream(ctx, req) {
			failure = err
			if !yield(ev, err) || err != nil {
				return
			}
		}
	}
}
