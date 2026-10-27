package llm

import (
	"context"
	"errors"
	"fmt"
	"iter"
	"sync"
)

// Price is a model's list price in USD per million tokens (MTok).
type Price struct {
	Input      float64
	Output     float64
	CacheWrite float64 // 5-minute TTL write: 1.25 × Input
	CacheRead  float64
}

// Prices is a snapshot of Anthropic first-party list prices (September 2026).
// Prices change — check the pricing page before relying on these numbers.
var Prices = map[string]Price{
	"claude-fable-5-1":  {Input: 10, Output: 50, CacheWrite: 12.5, CacheRead: 0.25},
	"claude-opus-5-5":   {Input: 4, Output: 20, CacheWrite: 5, CacheRead: 0.20},
	"claude-sonnet-5-5": {Input: 2, Output: 10, CacheWrite: 2.5, CacheRead: 0.20},
	"claude-haiku-4-5":  {Input: 1, Output: 5, CacheWrite: 1.25, CacheRead: 0.10},
}

// Cost returns the USD cost of one call:
// (in·P_in + out·P_out + write·P_write + read·P_read) / 10^6.
func (p Price) Cost(u Usage) float64 {
	return (float64(u.InputTokens)*p.Input + float64(u.OutputTokens)*p.Output +
		float64(u.CacheCreationInputTokens)*p.CacheWrite + float64(u.CacheReadInputTokens)*p.CacheRead) / 1e6
}

// ErrBudgetExceeded is returned before a call once the Meter's budget is spent.
var ErrBudgetExceeded = errors.New("llm: budget exceeded")

// Meter accumulates usage and cost per model; safe for concurrent use.
type Meter struct {
	Budget float64 // USD; 0 means unlimited

	mu    sync.Mutex
	usage map[string]Usage
	cost  float64
	calls int
}

func (m *Meter) Record(model string, u Usage) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.usage == nil {
		m.usage = make(map[string]Usage)
	}
	m.usage[model] = m.usage[model].Add(u)
	m.cost += Prices[model].Cost(u)
	m.calls++
}

// Spent returns the total cost in USD so far.
func (m *Meter) Spent() float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.cost
}

// Usage returns the accumulated usage of one model.
func (m *Meter) Usage(model string) Usage {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.usage[model]
}

func (m *Meter) String() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return fmt.Sprintf("%d calls, $%.4f", m.calls, m.cost)
}

func (m *Meter) check() error {
	if m.Budget > 0 && m.Spent() >= m.Budget {
		return fmt.Errorf("%w: spent $%.4f of $%.4f", ErrBudgetExceeded, m.Spent(), m.Budget)
	}
	return nil
}

// WithMeter records the usage of every call in m and refuses calls once m.Budget is spent.
// The check runs before the call, so the last call may overshoot the budget by its own cost.
func WithMeter(c Client, m *Meter) Client { return &metered{next: c, m: m} }

type metered struct {
	next Client
	m    *Meter
}

func (w *metered) Complete(ctx context.Context, req Request) (*Response, error) {
	if err := w.m.check(); err != nil {
		return nil, err
	}
	resp, err := w.next.Complete(ctx, req)
	if err == nil {
		w.m.Record(req.Model, resp.Usage)
	}
	return resp, err
}

func (w *metered) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		if err := w.m.check(); err != nil {
			yield(Event{}, err)
			return
		}
		var u Usage
		defer func() { w.m.Record(req.Model, u) }() // partial streams are billed too
		for ev, err := range w.next.Stream(ctx, req) {
			if err == nil && (ev.Type == EventStart || ev.Type == EventStop) {
				u = u.Add(ev.Usage)
			}
			if !yield(ev, err) || err != nil {
				return
			}
		}
	}
}
