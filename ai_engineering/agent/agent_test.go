package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"iter"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"sync"
	"testing"

	"algorithms_and_data_structures/ai_engineering/evals"
	"algorithms_and_data_structures/ai_engineering/llm"
)

// fakeLLM is a deterministic llm.Client: fn decides the response from the request.
// It records every request, so tests can assert what the model was shown.
type fakeLLM struct {
	mu   sync.Mutex
	fn   func(req llm.Request) (*llm.Response, error)
	reqs []llm.Request
}

func (f *fakeLLM) Complete(ctx context.Context, req llm.Request) (*llm.Response, error) {
	f.mu.Lock()
	f.reqs = append(f.reqs, req)
	f.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return f.fn(req)
}

func (f *fakeLLM) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		resp, err := f.Complete(ctx, req)
		if err != nil {
			yield(llm.Event{}, err)
			return
		}
		if !yield(llm.Event{Type: llm.EventStart, Usage: llm.Usage{InputTokens: resp.Usage.InputTokens}}, nil) ||
			!yield(llm.Event{Type: llm.EventTextDelta, Text: resp.Text}, nil) {
			return
		}
		for _, c := range resp.ToolCalls {
			if !yield(llm.Event{Type: llm.EventToolCall, ToolCall: c}, nil) {
				return
			}
		}
		yield(llm.Event{Type: llm.EventStop, StopReason: resp.StopReason, Usage: llm.Usage{OutputTokens: resp.Usage.OutputTokens}}, nil)
	}
}

func (f *fakeLLM) requests() []llm.Request {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]llm.Request(nil), f.reqs...)
}

// scripted returns responses in order; running out is a test failure.
func scripted(t *testing.T, resps ...*llm.Response) *fakeLLM {
	t.Helper()
	i := 0
	return &fakeLLM{fn: func(llm.Request) (*llm.Response, error) {
		if i >= len(resps) {
			t.Errorf("unexpected model call #%d", i+1)
			return nil, errors.New("script exhausted")
		}
		i++
		return resps[i-1], nil
	}}
}

func call(id, name, input string) llm.ToolCall {
	return llm.ToolCall{ID: id, Name: name, Input: json.RawMessage(input)}
}

func toolUse(text string, calls ...llm.ToolCall) *llm.Response {
	return &llm.Response{Text: text, ToolCalls: calls, StopReason: "tool_use", Usage: llm.Usage{InputTokens: 100, OutputTokens: 20}}
}

func final(text string) *llm.Response {
	return &llm.Response{Text: text, StopReason: "end_turn", Usage: llm.Usage{InputTokens: 150, OutputTokens: 30}}
}

func TestSchemaOf(t *testing.T) {
	type in struct {
		Query string   `json:"query" desc:"what to find"`
		K     int      `json:"k,omitempty"`
		Mode  string   `json:"mode" enum:"fast,exact"`
		Tags  []string `json:"tags,omitempty"`
		Score *float64 `json:"score"`
		skip  int
		Dash  int `json:"-"`
	}
	got := SchemaOf(reflect.TypeOf(in{}))
	want := map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query": map[string]any{"type": "string", "description": "what to find"},
			"k":     map[string]any{"type": "integer"},
			"mode":  map[string]any{"type": "string", "enum": []string{"fast", "exact"}},
			"tags":  map[string]any{"type": "array", "items": map[string]any{"type": "string"}},
			"score": map[string]any{"type": "number"},
		},
		"required":             []string{"query", "mode"},
		"additionalProperties": false,
	}
	if !reflect.DeepEqual(got, want) {
		g, _ := json.MarshalIndent(got, "", " ")
		t.Fatalf("schema:\n%s", g)
	}
}

func TestToolInputValidation(t *testing.T) {
	type in struct {
		Expression string `json:"expression"`
		Mode       string `json:"mode,omitempty" enum:"deg,rad"`
	}
	tool := NewTool("calc", "", func(_ context.Context, in in) (string, error) { return in.Expression, nil })
	cases := []struct{ input, wantErr string }{
		{`{"expression":"1+1"}`, ""},
		{`{}`, `missing required field "expression"`},
		{`{"expression":"1","extra":1}`, `unknown field "extra"`},
		{`{"expression":42}`, "cannot unmarshal number"},
		{`{"expression":"1","mode":"grad"}`, `must be one of [deg rad]`},
		{`[1,2]`, "must be a JSON object"},
	}
	for _, c := range cases {
		out, err := tool.Call(context.Background(), json.RawMessage(c.input))
		if c.wantErr == "" {
			if err != nil || out != "1+1" {
				t.Errorf("%s: %q %v", c.input, out, err)
			}
		} else if err == nil || !strings.Contains(err.Error(), c.wantErr) {
			t.Errorf("%s: err = %v, want %q", c.input, err, c.wantErr)
		}
	}
}

func TestRegistry(t *testing.T) {
	r := NewRegistry(Calculator())
	if err := r.Register(Calculator()); err == nil {
		t.Fatal("duplicate name must be rejected")
	}
	if err := r.Register(NewTool("bad name!", "", func(context.Context, CalculatorInput) (string, error) { return "", nil })); err == nil {
		t.Fatal("invalid name must be rejected")
	}
	panicky := NewTool("boom", "", func(context.Context, CalculatorInput) (string, error) { panic("oops") })
	r.Register(panicky)
	if res := r.Call(context.Background(), call("1", "boom", `{"expression":""}`)); !res.IsError || !strings.Contains(res.Content, "panicked") {
		t.Fatalf("panic must become an error result: %+v", res)
	}
	if res := r.Call(context.Background(), call("2", "nope", `{}`)); !res.IsError || res.CallID != "2" {
		t.Fatalf("unknown tool: %+v", res)
	}
	if names := r.Specs(); len(names) != 2 || names[0].Name != "calculator" || names[1].Name != "boom" {
		t.Fatalf("specs must keep registration order: %+v", names)
	}
}

func TestEval(t *testing.T) {
	cases := map[string]float64{
		"17*23":              391,
		"2+3*4":              14,
		"(2+3)*4":            20,
		"2^3^2":              512, // right-associative
		"-2^2":               -4,  // unary minus binds looser than ^
		"10 % 4":             2,
		"sqrt(16) + abs(-3)": 7,
		"1.5e3 / 3":          500,
		"2*pi":               6.283185307179586,
		"ln(e)":              1,
	}
	for expr, want := range cases {
		got, err := Eval(expr)
		if err != nil || got != want {
			t.Errorf("Eval(%q) = %v, %v; want %v", expr, got, err, want)
		}
	}
	for _, bad := range []string{"1/0", "2+", "foo(1)", "(1+2", "1 2", ""} {
		if _, err := Eval(bad); err == nil {
			t.Errorf("Eval(%q) must fail", bad)
		}
	}
}

func TestAgentLoop(t *testing.T) {
	f := scripted(t,
		toolUse("Посчитаю.", call("toolu_1", "calculator", `{"expression":"17*23"}`)),
		final("17 × 23 = 391"),
	)
	tr := &evals.Tracer{}
	a := &Agent{Client: f, Model: "m", System: "sys", Tools: NewRegistry(Calculator()), Tracer: tr}
	res, err := a.Run(context.Background(), "Сколько будет 17*23?")
	if err != nil {
		t.Fatal(err)
	}
	if res.Text != "17 × 23 = 391" || res.Steps != 2 || res.ToolCalls != 1 || res.Stop != StopDone {
		t.Fatalf("result = %+v", res)
	}
	if want := (llm.Usage{InputTokens: 250, OutputTokens: 50}); res.Usage != want {
		t.Fatalf("usage = %+v", res.Usage)
	}
	// user → assistant(tool_use) → user(tool_result) → assistant
	if len(res.Messages) != 4 || res.Messages[2].ToolResults[0].Content != "391" || res.Messages[2].ToolResults[0].CallID != "toolu_1" {
		t.Fatalf("messages = %+v", res.Messages)
	}
	reqs := f.requests()
	if len(reqs[0].Tools) != 1 || reqs[0].Tools[0].Name != "calculator" || reqs[1].Messages[2].ToolResults[0].Content != "391" {
		t.Fatalf("second request must carry the tool result: %+v", reqs[1].Messages)
	}
	var names []string
	for _, s := range tr.Spans() {
		names = append(names, s.Name)
	}
	if got := strings.Join(names, ","); got != "llm.call,tool.calculator,llm.call,agent.run" {
		t.Fatalf("spans = %s", got)
	}
}

func TestAgentParallelCallsOneMessage(t *testing.T) {
	f := scripted(t,
		toolUse("", call("a", "calculator", `{"expression":"1+1"}`), call("b", "calculator", `{"expression":"2*3"}`), call("c", "nope", `{}`)),
		final("2 и 6"),
	)
	a := &Agent{Client: f, Tools: NewRegistry(Calculator())}
	res, err := a.Run(context.Background(), "q")
	if err != nil {
		t.Fatal(err)
	}
	results := res.Messages[2].ToolResults
	if len(results) != 3 || results[0].Content != "2" || results[1].Content != "6" || !results[2].IsError {
		t.Fatalf("all results must come back in one message, in call order: %+v", results)
	}
}

func TestAgentStops(t *testing.T) {
	loop := &fakeLLM{fn: func(llm.Request) (*llm.Response, error) {
		return toolUse("", call("x", "calculator", `{"expression":"1"}`)), nil
	}}
	a := &Agent{Client: loop, Tools: NewRegistry(Calculator()), MaxSteps: 3}
	res, err := a.Run(context.Background(), "q")
	if !errors.Is(err, ErrMaxSteps) || res.Steps != 3 || res.Stop != StopMaxSteps {
		t.Fatalf("max steps: %v %+v", err, res)
	}

	// Each call costs 120 tokens; MaxTokens = 100 is the worst-case output of the next call.
	a = &Agent{Client: loop, Tools: NewRegistry(Calculator()), MaxTokens: 100, TokenBudget: 300}
	res, err = a.Run(context.Background(), "q")
	if !errors.Is(err, ErrTokenBudget) || res.Steps != 1 || res.Stop != StopBudget {
		// after 1 call: 120 used + (~prompt + 100) > 300 → stop before the 2nd call
		t.Fatalf("token budget: %v steps=%d", err, res.Steps)
	}

	trunc := scripted(t, &llm.Response{StopReason: "max_tokens", ToolCalls: []llm.ToolCall{call("t", "write_file", `{"path":"a`)}})
	res, err = (&Agent{Client: trunc, Tools: NewRegistry(Calculator())}).Run(context.Background(), "q")
	if !errors.Is(err, ErrTruncated) || res.ToolCalls != 0 {
		t.Fatalf("truncated tool call must not run: %v %+v", err, res)
	}

	refuse := scripted(t, &llm.Response{StopReason: "refusal"})
	if _, err := (&Agent{Client: refuse}).Run(context.Background(), "q"); !errors.Is(err, ErrRefusal) {
		t.Fatalf("refusal: %v", err)
	}
}

func TestDangerousToolNeedsConfirmation(t *testing.T) {
	dir := t.TempDir()
	sb, _ := NewSandbox(dir)
	write := call("w", "write_file", `{"path":"out.txt","content":"hi"}`)
	for _, tc := range []struct {
		name    string
		confirm ConfirmFunc
		written bool
		msg     string
	}{
		{"no hook denies", nil, false, "requires human confirmation"},
		{"user says no", func(context.Context, llm.ToolCall) (bool, error) { return false, nil }, false, "declined"},
		{"user says yes", func(_ context.Context, c llm.ToolCall) (bool, error) { return c.Name == "write_file", nil }, true, "wrote 2 bytes"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			f := scripted(t, toolUse("", write), final("ok"))
			a := &Agent{Client: f, Tools: NewRegistry(ProjectTools(sb, true)...), Confirm: tc.confirm}
			res, err := a.Run(context.Background(), "write")
			if err != nil {
				t.Fatal(err)
			}
			if got := res.Messages[2].ToolResults[0].Content; !strings.Contains(got, tc.msg) {
				t.Fatalf("result = %q, want %q", got, tc.msg)
			}
			_, statErr := sb.ReadFile("out.txt")
			if (statErr == nil) != tc.written {
				t.Fatalf("written = %v, want %v", statErr == nil, tc.written)
			}
			os.Remove(filepath.Join(dir, "out.txt"))
		})
	}
}

func TestTruncate(t *testing.T) {
	s := strings.Repeat("я", 1000) // 2000 bytes
	out := truncate(s, 300)
	if len(out) > 400 || !strings.Contains(out, "обрезано") || !strings.HasPrefix(out, "я") || !strings.HasSuffix(out, "я") {
		t.Fatalf("truncate: %d bytes %q…", len(out), out[:20])
	}
	if truncate("short", 300) != "short" {
		t.Fatal("short strings are unchanged")
	}
}

func ExampleAgent() {
	// A scripted model: first it asks for the calculator, then it answers.
	model := &fakeLLM{}
	step := 0
	model.fn = func(req llm.Request) (*llm.Response, error) {
		step++
		if step == 1 {
			return &llm.Response{StopReason: "tool_use", ToolCalls: []llm.ToolCall{
				{ID: "toolu_1", Name: "calculator", Input: json.RawMessage(`{"expression":"17*23"}`)}}}, nil
		}
		last := req.Messages[len(req.Messages)-1].ToolResults[0]
		return &llm.Response{StopReason: "end_turn", Text: "Ответ: " + last.Content}, nil
	}
	a := &Agent{Client: model, Tools: NewRegistry(Calculator())}
	res, _ := a.Run(context.Background(), "Сколько будет 17*23?")
	fmt.Println(res.Text, res.Steps)
	// Output: Ответ: 391 2
}
