// Package agent is a tool-using LLM agent (lectures L18 and L20): a tool registry with
// JSON Schemas generated from Go structs, the think → act → observe loop with step and
// token budgets, built-in sandboxed tools, a confirmation hook for dangerous tools,
// a context manager (token budget, tool-result clearing, compaction), orchestrator-workers
// on goroutines and a needle-in-a-haystack harness. Subpackage mcp exposes a Registry
// as a Model Context Protocol server.
package agent

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"unicode/utf8"

	"algorithms_and_data_structures/ai_engineering/evals"
	"algorithms_and_data_structures/ai_engineering/llm"
)

// ConfirmFunc asks a human (or a policy) whether a dangerous tool call may run.
type ConfirmFunc func(ctx context.Context, call llm.ToolCall) (bool, error)

// Agent runs the tool loop: call the model; if it asked for tools, run them and send
// the results back; repeat until it answers without tools or a budget runs out.
type Agent struct {
	Client llm.Client
	Model  string
	System string
	Tools  *Registry
	Effort string // reasoning effort for every call; "" = provider default

	MaxSteps      int // model calls per run; default 20
	MaxTokens     int // output tokens per call; default 4096
	TokenBudget   int // total tokens (input + output, all calls); 0 = unlimited
	MaxToolOutput int // bytes of one tool result kept in the context; default 16 KiB

	// Confirm approves dangerous tools. nil denies them all: least privilege by default.
	Confirm ConfirmFunc
	// Context, when set, keeps the prompt under a token budget (L20).
	Context *ContextManager
	// Tracer, when set, records spans agent.run → llm.call / tool.<name>.
	Tracer *evals.Tracer
}

// Stop says why a run ended.
type Stop string

const (
	StopDone      Stop = "done"         // the model answered without calling tools
	StopMaxSteps  Stop = "max_steps"    // step budget exhausted
	StopBudget    Stop = "token_budget" // the next call could exceed TokenBudget
	StopRefusal   Stop = "refusal"      // the model declined
	StopTruncated Stop = "max_tokens"   // a tool call was cut off by MaxTokens
)

var (
	ErrMaxSteps    = errors.New("agent: step budget exhausted")
	ErrTokenBudget = errors.New("agent: token budget exhausted")
	ErrRefusal     = errors.New("agent: model refused")
	ErrTruncated   = errors.New("agent: response truncated by max_tokens during a tool call")
)

// Result of a run. Messages is the full history, so a run can be continued
// with RunMessages(append(res.Messages, nextUserMessage)).
type Result struct {
	Text      string
	Messages  []llm.Message
	Steps     int
	ToolCalls int
	Usage     llm.Usage
	Stop      Stop
}

// TotalTokens is everything billed: input (incl. cache reads/writes) and output.
func TotalTokens(u llm.Usage) int {
	return u.InputTokens + u.CacheCreationInputTokens + u.CacheReadInputTokens + u.OutputTokens
}

// Run starts a new conversation with one user message.
func (a *Agent) Run(ctx context.Context, task string) (*Result, error) {
	return a.RunMessages(ctx, []llm.Message{{Role: llm.RoleUser, Content: task}})
}

// Target adapts the agent to evals.Runner (L19): input → final answer.
func (a *Agent) Target() evals.Target {
	return func(ctx context.Context, input string) (string, error) {
		res, err := a.Run(ctx, input)
		if res == nil {
			return "", err
		}
		return res.Text, err
	}
}

// RunMessages continues a conversation that ends with a user message.
// On a budget stop it returns the partial result together with the error.
func (a *Agent) RunMessages(ctx context.Context, history []llm.Message) (res *Result, err error) {
	maxSteps := a.MaxSteps
	if maxSteps <= 0 {
		maxSteps = 20
	}
	maxTokens := a.MaxTokens
	if maxTokens <= 0 {
		maxTokens = 4096
	}
	ctx, end := a.span(ctx, "agent.run")
	res = &Result{Messages: append([]llm.Message(nil), history...)}
	defer func() {
		end(err, "model", a.Model, "steps", res.Steps, "tool_calls", res.ToolCalls,
			"total_tokens", TotalTokens(res.Usage), "stop", string(res.Stop))
	}()

	for {
		if err := ctx.Err(); err != nil {
			return res, err
		}
		if res.Steps >= maxSteps {
			res.Stop = StopMaxSteps
			return res, ErrMaxSteps
		}
		req := llm.Request{Model: a.Model, System: a.System, Tools: a.Tools.Specs(), MaxTokens: maxTokens, Effort: a.Effort}
		if a.Context != nil {
			req.Messages = res.Messages
			msgs, rep, err := a.Context.Fit(ctx, req)
			res.Usage = res.Usage.Add(rep.Usage)
			if err != nil {
				return res, fmt.Errorf("agent: context: %w", err)
			}
			res.Messages = msgs
		}
		req.Messages = res.Messages
		// Worst case of the next call: its whole prompt plus MaxTokens of output.
		if a.TokenBudget > 0 && TotalTokens(res.Usage)+int(llm.EstimateTokens(req)) > a.TokenBudget {
			res.Stop = StopBudget
			return res, ErrTokenBudget
		}

		res.Steps++
		callCtx, endCall := a.span(ctx, "llm.call")
		resp, err := a.Client.Complete(callCtx, req)
		if err != nil {
			endCall(err, "step", res.Steps)
			return res, err
		}
		endCall(nil, "step", res.Steps, "input_tokens", resp.Usage.InputTokens+resp.Usage.CacheReadInputTokens+resp.Usage.CacheCreationInputTokens,
			"output_tokens", resp.Usage.OutputTokens, "stop_reason", resp.StopReason, "tool_calls", len(resp.ToolCalls))
		res.Usage = res.Usage.Add(resp.Usage)
		res.Text = resp.Text
		res.Messages = append(res.Messages, llm.Message{Role: llm.RoleAssistant, Content: resp.Text, ToolCalls: resp.ToolCalls, Raw: resp.Raw})

		switch {
		case resp.StopReason == "refusal":
			res.Stop = StopRefusal
			return res, ErrRefusal
		case len(resp.ToolCalls) == 0:
			res.Stop = StopDone
			return res, nil
		case resp.StopReason == "max_tokens" || resp.StopReason == "length":
			// The last tool input may be cut mid-JSON: never execute it.
			res.Stop = StopTruncated
			return res, ErrTruncated
		}

		results := a.runTools(ctx, resp.ToolCalls)
		res.ToolCalls += len(results)
		// All results of one turn go back in ONE user message, in call order.
		res.Messages = append(res.Messages, llm.Message{Role: llm.RoleUser, ToolResults: results})
	}
}

// runTools asks Confirm for dangerous calls (sequentially — a human answers one at a time),
// then runs the approved calls concurrently and returns results in call order.
func (a *Agent) runTools(ctx context.Context, calls []llm.ToolCall) []llm.ToolResult {
	results := make([]llm.ToolResult, len(calls))
	approved := make([]bool, len(calls))
	for i, c := range calls {
		approved[i] = true
		t, ok := a.Tools.Get(c.Name)
		if !ok || !IsDangerous(t) {
			continue
		}
		if a.Confirm == nil {
			approved[i] = false
			results[i] = llm.ToolResult{CallID: c.ID, IsError: true,
				Content: fmt.Sprintf("tool %q requires human confirmation and none is configured; do not retry it", c.Name)}
			continue
		}
		ok, err := a.Confirm(ctx, c)
		if err != nil || !ok {
			approved[i] = false
			results[i] = llm.ToolResult{CallID: c.ID, IsError: true, Content: "the user declined this action"}
		}
	}
	var wg sync.WaitGroup
	for i, c := range calls {
		if !approved[i] {
			continue
		}
		wg.Go(func() {
			tctx, end := a.span(ctx, "tool."+c.Name)
			r := a.Tools.Call(tctx, c)
			r.Content = truncate(r.Content, a.maxToolOutput())
			var err error
			if r.IsError {
				err = errors.New(r.Content)
			}
			end(err, "bytes", len(r.Content))
			results[i] = r
		})
	}
	wg.Wait()
	return results
}

func (a *Agent) maxToolOutput() int {
	if a.MaxToolOutput > 0 {
		return a.MaxToolOutput
	}
	return 16 << 10
}

// truncate keeps the head and the tail of a long tool output (errors are usually at
// the end of a log, the context at the start) and says how much was dropped.
func truncate(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	head, tail := limit*2/3, limit/3
	for head > 0 && !utf8.RuneStart(s[head]) {
		head--
	}
	cut := len(s) - tail
	for cut < len(s) && !utf8.RuneStart(s[cut]) {
		cut++
	}
	return fmt.Sprintf("%s\n…[обрезано %d байт]…\n%s", s[:head], cut-head, s[cut:])
}

// span starts a tracing span if a Tracer is set; end records attributes as key, value pairs.
func (a *Agent) span(ctx context.Context, name string) (context.Context, func(err error, kv ...any)) {
	return startSpan(a.Tracer, ctx, name)
}

func startSpan(tr *evals.Tracer, ctx context.Context, name string) (context.Context, func(err error, kv ...any)) {
	if tr == nil {
		return ctx, func(error, ...any) {}
	}
	ctx, s := tr.Start(ctx, name)
	return ctx, func(err error, kv ...any) {
		for i := 0; i+1 < len(kv); i += 2 {
			s.SetAttr(fmt.Sprint(kv[i]), kv[i+1])
		}
		s.Finish(err)
	}
}
