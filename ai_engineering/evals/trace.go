package evals

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"time"
)

// Span is one timed operation (an LLM call, a retrieval, a tool call), in the spirit
// of OpenTelemetry: spans form a tree via ParentID and share a TraceID per request.
type Span struct {
	TraceID  string         `json:"trace_id"`
	SpanID   string         `json:"span_id"`
	ParentID string         `json:"parent_id,omitempty"`
	Name     string         `json:"name"`
	Start    time.Time      `json:"start"`
	End      time.Time      `json:"end"`
	Attrs    map[string]any `json:"attrs,omitempty"`
	Error    string         `json:"error,omitempty"`

	mu     sync.Mutex
	tracer *Tracer
}

// SetAttr records a key/value, e.g. "model", "input_tokens", "cost_usd".
func (s *Span) SetAttr(key string, v any) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.Attrs == nil {
		s.Attrs = map[string]any{}
	}
	s.Attrs[key] = v
}

// Finish stops the clock and hands the span to its tracer. err may be nil.
func (s *Span) Finish(err error) {
	s.mu.Lock()
	s.End = time.Now()
	if err != nil {
		s.Error = err.Error()
	}
	s.mu.Unlock()
	s.tracer.record(s)
}

// Duration is End − Start.
func (s *Span) Duration() time.Duration { return s.End.Sub(s.Start) }

// Tracer collects finished spans in memory.
type Tracer struct {
	next  atomic.Uint64
	mu    sync.Mutex
	spans []*Span
}

type spanKey struct{}

// Start begins a span; if ctx already carries a span, the new one becomes its child.
// The returned context carries the new span.
func (t *Tracer) Start(ctx context.Context, name string) (context.Context, *Span) {
	s := &Span{Name: name, Start: time.Now(), SpanID: t.id(), tracer: t}
	if parent, ok := ctx.Value(spanKey{}).(*Span); ok {
		s.TraceID, s.ParentID = parent.TraceID, parent.SpanID
	} else {
		s.TraceID = t.id()
	}
	return context.WithValue(ctx, spanKey{}, s), s
}

func (t *Tracer) id() string { return fmt.Sprintf("%016x", t.next.Add(1)) }

func (t *Tracer) record(s *Span) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.spans = append(t.spans, s)
}

// Spans returns the finished spans in finishing order.
func (t *Tracer) Spans() []*Span {
	t.mu.Lock()
	defer t.mu.Unlock()
	return append([]*Span(nil), t.spans...)
}

// ExportJSON writes one JSON object per finished span (JSON Lines).
func (t *Tracer) ExportJSON(w io.Writer) error {
	enc := json.NewEncoder(w)
	for _, s := range t.Spans() {
		s.mu.Lock()
		err := enc.Encode(s)
		s.mu.Unlock()
		if err != nil {
			return err
		}
	}
	return nil
}
