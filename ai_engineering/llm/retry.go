package llm

import (
	"context"
	"errors"
	"iter"
	"math/rand/v2"
	"time"
)

// RetryPolicy configures exponential backoff with full jitter (AWS Architecture Blog, 2015):
//
//	sleep_n = random(0, min(Max, Base · 2^n))   for attempt n = 0, 1, 2, ...
//
// and never less than the server's retry-after hint.
type RetryPolicy struct {
	MaxAttempts int // total attempts including the first one
	Base        time.Duration
	Max         time.Duration

	Rand  func() float64                                   // uniform [0,1); defaults to math/rand/v2
	Sleep func(ctx context.Context, d time.Duration) error // defaults to a context-aware timer
}

// DefaultRetryPolicy: 4 attempts, 0.5 s base, 30 s cap.
var DefaultRetryPolicy = RetryPolicy{MaxAttempts: 4, Base: 500 * time.Millisecond, Max: 30 * time.Second}

// Backoff returns the delay before retry number n (n = 0 for the first retry).
func (p RetryPolicy) Backoff(n int, retryAfter time.Duration) time.Duration {
	ceiling := p.Base << n
	if ceiling > p.Max || ceiling <= 0 { // <= 0 guards the shift overflow
		ceiling = p.Max
	}
	r := rand.Float64
	if p.Rand != nil {
		r = p.Rand
	}
	d := time.Duration(r() * float64(ceiling))
	return max(d, retryAfter)
}

func (p RetryPolicy) sleep(ctx context.Context, d time.Duration) error {
	if p.Sleep != nil {
		return p.Sleep(ctx, d)
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// WithRetry wraps c so that retryable failures are repeated according to p.
func WithRetry(c Client, p RetryPolicy) Client { return &retrying{next: c, policy: p} }

type retrying struct {
	next   Client
	policy RetryPolicy
}

func (r *retrying) wait(ctx context.Context, attempt int, err error) error {
	var retryAfter time.Duration
	if apiErr, ok := errors.AsType[*APIError](err); ok {
		retryAfter = apiErr.RetryAfter
	}
	return r.policy.sleep(ctx, r.policy.Backoff(attempt, retryAfter))
}

func (r *retrying) Complete(ctx context.Context, req Request) (*Response, error) {
	for attempt := 0; ; attempt++ {
		resp, err := r.next.Complete(ctx, req)
		if err == nil || !IsRetryable(err) || attempt+1 >= r.policy.MaxAttempts {
			return resp, err
		}
		if werr := r.wait(ctx, attempt, err); werr != nil {
			return nil, werr
		}
	}
}

// Stream retries only failures that happen before the first event reached the caller.
// After text has been shown to a user, silently restarting would duplicate it,
// so a mid-stream failure is returned as is and the caller decides.
func (r *retrying) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		for attempt := 0; ; attempt++ {
			delivered := false
			var failure error
			for ev, err := range r.next.Stream(ctx, req) {
				if err != nil {
					failure = err
					break
				}
				delivered = true
				if !yield(ev, nil) {
					return
				}
			}
			if failure == nil {
				return
			}
			if delivered || !IsRetryable(failure) || attempt+1 >= r.policy.MaxAttempts {
				yield(Event{}, failure)
				return
			}
			if werr := r.wait(ctx, attempt, failure); werr != nil {
				yield(Event{}, werr)
				return
			}
		}
	}
}
