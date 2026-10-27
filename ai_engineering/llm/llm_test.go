package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestReadSSE(t *testing.T) {
	raw := ": keep-alive comment\r\n" +
		"event: message_start\r\n" +
		"data: {\"a\":1}\r\n\r\n" +
		"data: line1\n" +
		"data:line2\n\n" +
		"event: tail\n" +
		"data: no blank line at the end"
	var got []SSEEvent
	for ev, err := range ReadSSE(strings.NewReader(raw)) {
		if err != nil {
			t.Fatal(err)
		}
		got = append(got, ev)
	}
	want := []SSEEvent{
		{Event: "message_start", Data: `{"a":1}`},
		{Data: "line1\nline2"},
		{Event: "tail", Data: "no blank line at the end"},
	}
	if fmt.Sprint(got) != fmt.Sprint(want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func newAnthropic(url string) *Anthropic {
	return &Anthropic{APIKey: "test-key", BaseURL: url, Version: "2023-06-01", HTTP: http.DefaultClient}
}

func TestAnthropicComplete(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/messages" || r.Header.Get("x-api-key") != "test-key" || r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("bad request: %s %v", r.URL.Path, r.Header)
		}
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		system := body["system"].([]any)[0].(map[string]any)
		if system["cache_control"].(map[string]any)["type"] != "ephemeral" {
			t.Errorf("system prompt must carry cache_control: %v", system)
		}
		format := body["output_config"].(map[string]any)["format"].(map[string]any)
		if format["type"] != "json_schema" {
			t.Errorf("output_config.format = %v", format)
		}
		if _, ok := body["temperature"]; ok {
			t.Errorf("temperature must be omitted when not set")
		}
		io.WriteString(w, `{"id":"msg_1","model":"claude-opus-5-5","stop_reason":"end_turn",
			"content":[{"type":"thinking","thinking":""},{"type":"text","text":"{\"ok\":"},{"type":"text","text":"true}"}],
			"usage":{"input_tokens":10,"output_tokens":5,"cache_creation_input_tokens":2000,"cache_read_input_tokens":0}}`)
	}))
	defer srv.Close()

	resp, err := newAnthropic(srv.URL).Complete(context.Background(), Request{
		Model: "claude-opus-5-5", MaxTokens: 100, System: "You are terse.", CacheSystem: true,
		Messages:   []Message{{Role: RoleUser, Content: "hi"}},
		JSONSchema: map[string]any{"type": "object", "additionalProperties": false},
	})
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != `{"ok":true}` || resp.StopReason != "end_turn" || resp.Usage.CacheCreationInputTokens != 2000 {
		t.Fatalf("resp = %+v", resp)
	}
}

// writeSSE writes events and flushes after each so the client sees them incrementally.
func writeSSE(w http.ResponseWriter, events ...string) {
	w.Header().Set("content-type", "text/event-stream")
	for _, e := range events {
		io.WriteString(w, e+"\n\n")
		w.(http.Flusher).Flush()
	}
}

const anthropicStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_1","model":"claude-opus-5-5","usage":{"input_tokens":25,"output_tokens":1}}}|event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}|event: ping
data: {"type":"ping"}|event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}|event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":" world"}}|event: content_block_stop
data: {"type":"content_block_stop","index":0}|event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":12}}|event: message_stop
data: {"type":"message_stop"}`

func TestAnthropicStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body)
		if body["stream"] != true {
			t.Errorf("stream flag missing")
		}
		writeSSE(w, strings.Split(anthropicStream, "|")...)
	}))
	defer srv.Close()

	resp, err := Collect(newAnthropic(srv.URL).Stream(context.Background(), Request{Model: "claude-opus-5-5", MaxTokens: 64}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Hello world" || resp.StopReason != "end_turn" || resp.ID != "msg_1" {
		t.Fatalf("resp = %+v", resp)
	}
	// input from message_start; output only from message_delta, whose count is cumulative
	// (message_start's output_tokens: 1 must not be added on top).
	if resp.Usage.InputTokens != 25 || resp.Usage.OutputTokens != 12 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
}

func TestAnthropicStreamErrorEvent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, `event: error
data: {"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
	}))
	defer srv.Close()
	_, err := Collect(newAnthropic(srv.URL).Stream(context.Background(), Request{Model: "m", MaxTokens: 1}))
	apiErr, ok := errors.AsType[*APIError](err)
	if !ok || apiErr.StatusCode != 529 || !IsRetryable(err) {
		t.Fatalf("err = %v", err)
	}
}

func TestOpenAICompatible(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" || r.Header.Get("authorization") != "Bearer k" {
			t.Errorf("bad request %s %v", r.URL.Path, r.Header)
		}
		var body struct {
			Stream   bool         `json:"stream"`
			Messages []oaiMessage `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		if body.Messages[0].Role != "system" {
			t.Errorf("system prompt must be the first message: %v", body.Messages)
		}
		if !body.Stream {
			io.WriteString(w, `{"id":"c1","model":"gogpt","choices":[{"message":{"content":"hi"},"finish_reason":"stop"}],"usage":{"prompt_tokens":7,"completion_tokens":1}}`)
			return
		}
		writeSSE(w,
			`data: {"id":"c1","model":"gogpt","choices":[{"delta":{"content":"Hel"}}]}`,
			`data: {"id":"c1","model":"gogpt","choices":[{"delta":{"content":"lo"},"finish_reason":"stop"}]}`,
			`data: {"id":"c1","model":"gogpt","choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}`,
			`data: [DONE]`)
	}))
	defer srv.Close()

	c := &OpenAICompatible{APIKey: "k", BaseURL: srv.URL}
	req := Request{Model: "gogpt", System: "sys", Messages: []Message{{Role: RoleUser, Content: "q"}}}
	resp, err := c.Complete(context.Background(), req)
	if err != nil || resp.Text != "hi" || resp.StopReason != "stop" || resp.Usage.InputTokens != 7 {
		t.Fatalf("complete: %+v %v", resp, err)
	}
	resp, err = Collect(c.Stream(context.Background(), req))
	if err != nil || resp.Text != "Hello" || resp.StopReason != "stop" || resp.Usage.OutputTokens != 2 {
		t.Fatalf("stream: %+v %v", resp, err)
	}
}

// noSleep records requested delays instead of sleeping.
func noSleep(delays *[]time.Duration) func(context.Context, time.Duration) error {
	return func(_ context.Context, d time.Duration) error {
		*delays = append(*delays, d)
		return nil
	}
}

func TestRetryOn429And529(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch calls.Add(1) {
		case 1:
			w.Header().Set("retry-after", "2")
			w.WriteHeader(http.StatusTooManyRequests)
			io.WriteString(w, `{"type":"error","error":{"type":"rate_limit_error","message":"slow down"}}`)
		case 2:
			w.WriteHeader(529)
			io.WriteString(w, `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`)
		default:
			io.WriteString(w, `{"id":"m","content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn","usage":{}}`)
		}
	}))
	defer srv.Close()

	var delays []time.Duration
	p := RetryPolicy{MaxAttempts: 4, Base: time.Second, Max: 10 * time.Second, Rand: func() float64 { return 0.5 }, Sleep: noSleep(&delays)}
	resp, err := WithRetry(newAnthropic(srv.URL), p).Complete(context.Background(), Request{Model: "m", MaxTokens: 1})
	if err != nil || resp.Text != "ok" || calls.Load() != 3 {
		t.Fatalf("resp=%v err=%v calls=%d", resp, err, calls.Load())
	}
	// Retry 0: jitter 0.5·1s = 0.5s, but retry-after says 2s → 2s. Retry 1: 0.5·2s = 1s.
	if len(delays) != 2 || delays[0] != 2*time.Second || delays[1] != time.Second {
		t.Fatalf("delays = %v", delays)
	}
}

func TestNoRetryOn400(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		calls.Add(1)
		w.WriteHeader(http.StatusBadRequest)
		io.WriteString(w, `{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens: required"}}`)
	}))
	defer srv.Close()
	var delays []time.Duration
	_, err := WithRetry(newAnthropic(srv.URL), RetryPolicy{MaxAttempts: 5, Base: time.Second, Max: time.Minute, Sleep: noSleep(&delays)}).
		Complete(context.Background(), Request{Model: "m"})
	apiErr, ok := errors.AsType[*APIError](err)
	if !ok || apiErr.Type != "invalid_request_error" || calls.Load() != 1 {
		t.Fatalf("err=%v calls=%d", err, calls.Load())
	}
}

func TestBackoffFullJitter(t *testing.T) {
	p := RetryPolicy{Base: 500 * time.Millisecond, Max: 4 * time.Second, Rand: func() float64 { return 0.999999 }}
	want := []time.Duration{500, 1000, 2000, 4000, 4000} // ceilings in ms: Base·2^n capped at Max
	for n, w := range want {
		got := p.Backoff(n, 0)
		if got > w*time.Millisecond || got < w*time.Millisecond*99/100 {
			t.Fatalf("n=%d: %v, want ≈%vms", n, got, w)
		}
	}
	p.Rand = func() float64 { return 0 }
	if p.Backoff(3, 0) != 0 {
		t.Fatal("full jitter must allow zero delay")
	}
	if p.Backoff(60, 0) != 0 { // shift overflow must be capped, not negative
		t.Fatal("overflowed backoff")
	}
}

func TestStreamRetryOnlyBeforeFirstEvent(t *testing.T) {
	var calls atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if calls.Add(1) == 1 {
			w.WriteHeader(529)
			return
		}
		writeSSE(w, strings.Split(anthropicStream, "|")[:4]...) // start + one delta, then the connection drops
		panic(http.ErrAbortHandler)
	}))
	defer srv.Close()
	var delays []time.Duration
	c := WithRetry(newAnthropic(srv.URL), RetryPolicy{MaxAttempts: 5, Base: time.Millisecond, Max: time.Millisecond, Sleep: noSleep(&delays)})

	var text strings.Builder
	var streamErr error
	for ev, err := range c.Stream(context.Background(), Request{Model: "m", MaxTokens: 1}) {
		if err != nil {
			streamErr = err
			break
		}
		text.WriteString(ev.Text)
	}
	// First attempt (529 before any event) is retried; the mid-stream drop of the second is not.
	if calls.Load() != 2 || text.String() != "Hello" || streamErr == nil {
		t.Fatalf("calls=%d text=%q err=%v", calls.Load(), text.String(), streamErr)
	}
}

func TestCancellation(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, strings.Split(anthropicStream, "|")[0])
		select { // a slow model: nothing more until the client goes away
		case <-r.Context().Done():
		case <-release:
		}
	}))
	defer srv.Close()
	defer close(release)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	start := time.Now()
	var got error
	for ev, err := range newAnthropic(srv.URL).Stream(ctx, Request{Model: "m", MaxTokens: 1}) {
		if err != nil {
			got = err
			break
		}
		if ev.Type == EventStart {
			cancel() // e.g. the user closed the browser tab
		}
	}
	if !errors.Is(got, context.Canceled) || time.Since(start) > 5*time.Second {
		t.Fatalf("err=%v after %v", got, time.Since(start))
	}
	// A canceled request must not be retried.
	if IsRetryable(got) {
		t.Fatal("context.Canceled must not be retryable")
	}
}

type fakeClient struct {
	errs  []error
	calls int
	usage Usage
}

func (f *fakeClient) Complete(ctx context.Context, req Request) (*Response, error) {
	f.calls++
	if len(f.errs) > 0 {
		err := f.errs[0]
		f.errs = f.errs[1:]
		if err != nil {
			return nil, err
		}
	}
	return &Response{Text: "ok", Usage: f.usage}, nil
}

func (f *fakeClient) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		resp, err := f.Complete(ctx, req)
		if err != nil {
			yield(Event{}, err)
			return
		}
		_ = yield(Event{Type: EventStart, Usage: Usage{InputTokens: resp.Usage.InputTokens}}, nil) &&
			yield(Event{Type: EventTextDelta, Text: resp.Text}, nil) &&
			yield(Event{Type: EventStop, StopReason: "end_turn", Usage: Usage{OutputTokens: resp.Usage.OutputTokens}}, nil)
	}
}

func TestCost(t *testing.T) {
	// Opus 5.5: 10 000 in × $4/M + 1 000 out × $20/M = 0.04 + 0.02 = $0.06.
	if c := Prices["claude-opus-5-5"].Cost(Usage{InputTokens: 10_000, OutputTokens: 1_000}); math.Abs(c-0.06) > 1e-12 {
		t.Fatalf("cost = %v", c)
	}
	// 100k cached prefix read: 100 000 × $0.20/M = $0.02 instead of $0.40 uncached.
	if c := Prices["claude-opus-5-5"].Cost(Usage{CacheReadInputTokens: 100_000}); math.Abs(c-0.02) > 1e-12 {
		t.Fatalf("cache read cost = %v", c)
	}
}

func TestMeterBudget(t *testing.T) {
	f := &fakeClient{usage: Usage{InputTokens: 1_000_000}} // $4 per call on Opus 5.5
	m := &Meter{Budget: 10}
	c := WithMeter(f, m)
	req := Request{Model: "claude-opus-5-5"}
	for range 3 {
		if _, err := c.Complete(context.Background(), req); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := c.Complete(context.Background(), req); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("4th call err = %v", err)
	}
	if f.calls != 3 || math.Abs(m.Spent()-12) > 1e-9 {
		t.Fatalf("calls=%d spent=%v", f.calls, m.Spent())
	}
	if _, err := Collect(WithMeter(&fakeClient{usage: Usage{InputTokens: 10, OutputTokens: 20}}, m).Stream(context.Background(), req)); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("stream over budget err = %v", err)
	}
}

func TestTokenBucket(t *testing.T) {
	now := time.Unix(0, 0)
	b := NewTokenBucket(100, 600) // burst 100, 600/min = 10/s
	b.now = func() time.Time { return now }
	if !b.TryTake(100) || b.TryTake(1) {
		t.Fatal("bucket must allow exactly the burst")
	}
	now = now.Add(2 * time.Second) // +20 units
	if !b.TryTake(20) || b.TryTake(1) {
		t.Fatal("refill of 10/s for 2s must give 20 units")
	}
	now = now.Add(time.Hour)
	if !b.TryTake(100) || b.TryTake(1) {
		t.Fatal("refill must be capped at capacity")
	}
	if wait, _ := b.reserve(30); wait != 3*time.Second {
		t.Fatalf("wait for 30 units at 10/s = %v, want 3s", wait)
	}
	if _, err := b.reserve(101); err == nil {
		t.Fatal("request larger than capacity must fail")
	}
}

func TestTokenBucketWaitCanceled(t *testing.T) {
	b := NewTokenBucket(1, 1) // 1 per minute
	b.TryTake(1)
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if err := b.Wait(ctx, 1); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("err = %v", err)
	}
}

func TestBreaker(t *testing.T) {
	now := time.Unix(0, 0)
	b := NewBreaker(2, time.Minute)
	b.now = func() time.Time { return now }
	overloaded := &APIError{StatusCode: 529}
	f := &fakeClient{errs: []error{overloaded, overloaded, nil, overloaded}}
	c := WithBreaker(f, b)
	ctx := context.Background()

	c.Complete(ctx, Request{})
	c.Complete(ctx, Request{})
	if b.State() != Open {
		t.Fatal("two failures must open the breaker")
	}
	if _, err := c.Complete(ctx, Request{}); !errors.Is(err, ErrCircuitOpen) || f.calls != 2 {
		t.Fatalf("open breaker must fail fast: %v, calls=%d", err, f.calls)
	}
	now = now.Add(time.Minute)
	if b.State() != HalfOpen {
		t.Fatal("cooldown must lead to half-open")
	}
	if _, err := c.Complete(ctx, Request{}); err != nil || b.State() != Closed {
		t.Fatalf("successful probe must close: %v %v", err, b.State())
	}
	// A 400 is our bug, not provider trouble: it must not count.
	f.errs = []error{&APIError{StatusCode: 400}, &APIError{StatusCode: 400}, &APIError{StatusCode: 400}}
	for range 3 {
		c.Complete(ctx, Request{})
	}
	if b.State() != Closed {
		t.Fatal("non-retryable errors must not open the breaker")
	}
}

func TestPromptTemplate(t *testing.T) {
	p := NewPromptTemplate("summarize", "v2", "You summarize text.", "Summarize in {{.Lang}}:\n{{.Text}}")
	req, err := p.Render("claude-opus-5-5", 500, map[string]string{"Lang": "Russian", "Text": "Go is fun."})
	if err != nil || req.Messages[0].Content != "Summarize in Russian:\nGo is fun." || !req.CacheSystem {
		t.Fatalf("req=%+v err=%v", req, err)
	}
	if _, err := p.Render("m", 1, map[string]string{"Lang": "x"}); err == nil {
		t.Fatal("missing template key must be an error")
	}
	q := NewPromptTemplate("summarize", "v2", "You summarize text!", "Summarize in {{.Lang}}:\n{{.Text}}")
	if p.Fingerprint() == q.Fingerprint() || !strings.HasPrefix(p.Fingerprint(), "summarize@v2#") {
		t.Fatalf("fingerprints %s %s", p.Fingerprint(), q.Fingerprint())
	}
}

func TestModelRouter(t *testing.T) {
	var seen string
	inner := &recordingClient{seen: &seen}
	c := WithModelRouter(inner, func(r Request) string {
		if len(r.Messages[0].Content) < 20 {
			return "claude-haiku-4-5"
		}
		return ""
	})
	c.Complete(context.Background(), Request{Model: "claude-opus-5-5", Messages: []Message{{Content: "short"}}})
	if seen != "claude-haiku-4-5" {
		t.Fatalf("routed to %q", seen)
	}
	c.Complete(context.Background(), Request{Model: "claude-opus-5-5", Messages: []Message{{Content: strings.Repeat("long ", 10)}}})
	if seen != "claude-opus-5-5" {
		t.Fatalf("routed to %q", seen)
	}
}

type recordingClient struct{ seen *string }

func (r *recordingClient) Complete(ctx context.Context, req Request) (*Response, error) {
	*r.seen = req.Model
	return &Response{}, nil
}

func (r *recordingClient) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	*r.seen = req.Model
	return func(func(Event, error) bool) {}
}
