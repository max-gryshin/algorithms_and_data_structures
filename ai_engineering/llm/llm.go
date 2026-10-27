// Package llm is a small provider-agnostic client for chat LLM APIs (lecture L16):
// the Anthropic Messages API and any OpenAI-compatible server (including our own from L14),
// with streaming over SSE, retries with full-jitter backoff, token-bucket rate limiting,
// a circuit breaker, cost accounting and prompt templates.
//
// It is written on raw net/http on purpose, to show what happens on the wire.
// In production prefer the official SDK (github.com/anthropics/anthropic-sdk-go),
// which implements the same ideas and tracks API changes.
package llm

import (
	"context"
	"encoding/json"
	"iter"
	"strings"
)

type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// Message is one turn of the conversation. Content is plain text.
//
// Tool use (L18) adds two optional parts: an assistant turn may carry ToolCalls (the model
// asks us to run functions), and the next user turn carries ToolResults (what they returned).
type Message struct {
	Role        Role
	Content     string
	ToolCalls   []ToolCall   // assistant turns only
	ToolResults []ToolResult // user turns only; sent before Content

	// Raw is the provider-native content of an assistant turn exactly as it was received
	// (Anthropic: the "content" array, including thinking blocks with their signatures).
	// When set, the adapter that produced it replays it verbatim instead of rebuilding the
	// turn from Content and ToolCalls: reasoning models require thinking blocks to come
	// back unchanged during a tool loop. Clear it if you edit the turn.
	Raw json.RawMessage
}

// Request is the provider-neutral input of one model call.
type Request struct {
	Model     string
	System    string
	Messages  []Message
	MaxTokens int

	// Temperature is optional: some models (e.g. Claude Opus 5.5) reject sampling parameters.
	Temperature   *float64
	StopSequences []string

	// CacheSystem marks the system prompt as a prompt-caching breakpoint (Anthropic only).
	CacheSystem bool

	// JSONSchema, when set, asks for structured output that validates against this schema.
	JSONSchema map[string]any

	// Tools the model may call (L18). The model answers with Response.ToolCalls.
	Tools []ToolSpec

	// Effort is the reasoning-effort level ("low", "medium", "high", "xhigh", "max"; L20).
	// Empty means the provider default. Anthropic: output_config.effort;
	// OpenAI-compatible: reasoning_effort.
	Effort string
}

// Usage counts tokens of one call. Total prompt size is
// InputTokens + CacheCreationInputTokens + CacheReadInputTokens.
type Usage struct {
	InputTokens              int
	OutputTokens             int
	CacheCreationInputTokens int
	CacheReadInputTokens     int
}

// Add returns the field-wise sum.
func (u Usage) Add(o Usage) Usage {
	return Usage{
		InputTokens:              u.InputTokens + o.InputTokens,
		OutputTokens:             u.OutputTokens + o.OutputTokens,
		CacheCreationInputTokens: u.CacheCreationInputTokens + o.CacheCreationInputTokens,
		CacheReadInputTokens:     u.CacheReadInputTokens + o.CacheReadInputTokens,
	}
}

// Response is a complete (non-streamed or collected) model answer.
type Response struct {
	ID         string
	Model      string
	Text       string
	StopReason string // end_turn, tool_use, max_tokens, stop_sequence, refusal, ... (OpenAI: stop, length, tool_calls)
	Usage      Usage
	ToolCalls  []ToolCall      // functions the model wants us to run, in order
	Raw        json.RawMessage // provider-native content; see Message.Raw
}

type EventType int

const (
	EventStart     EventType = iota // response started; Usage carries input tokens if known
	EventTextDelta                  // Text holds the next piece of the answer
	EventStop                       // answer finished; StopReason, final Usage and Raw are set
	EventToolCall                   // ToolCall holds one complete tool call (input fully streamed)
)

// Event is one item of a streamed response.
type Event struct {
	Type       EventType
	ID         string
	Model      string
	Text       string
	StopReason string
	Usage      Usage
	ToolCall   ToolCall
	Raw        json.RawMessage
}

// Client is implemented by providers and by middlewares that wrap another Client.
type Client interface {
	Complete(ctx context.Context, req Request) (*Response, error)
	// Stream yields events until the answer ends. Iteration stops at the first error;
	// breaking out of the loop early closes the underlying HTTP connection.
	Stream(ctx context.Context, req Request) iter.Seq2[Event, error]
}

// Collect drains a stream into a Response.
func Collect(events iter.Seq2[Event, error]) (*Response, error) {
	var resp Response
	var sb strings.Builder
	for ev, err := range events {
		if err != nil {
			return nil, err
		}
		switch ev.Type {
		case EventStart:
			resp.ID, resp.Model = ev.ID, ev.Model
			resp.Usage = resp.Usage.Add(ev.Usage)
		case EventTextDelta:
			sb.WriteString(ev.Text)
		case EventToolCall:
			resp.ToolCalls = append(resp.ToolCalls, ev.ToolCall)
		case EventStop:
			resp.StopReason = ev.StopReason
			resp.Usage = resp.Usage.Add(ev.Usage)
			if ev.Raw != nil {
				resp.Raw = ev.Raw
			}
		}
	}
	resp.Text = sb.String()
	return &resp, nil
}
