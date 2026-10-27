package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"
)

// Anthropic talks to the Claude Messages API: POST {BaseURL}/v1/messages.
type Anthropic struct {
	APIKey  string
	BaseURL string // https://api.anthropic.com unless overridden (tests, proxies)
	Version string // anthropic-version header
	HTTP    *http.Client
}

// NewAnthropicFromEnv reads ANTHROPIC_API_KEY (and optionally ANTHROPIC_BASE_URL).
// Keys are never hardcoded: they live in the environment or a secret manager.
func NewAnthropicFromEnv() (*Anthropic, error) {
	key := os.Getenv("ANTHROPIC_API_KEY")
	if key == "" {
		return nil, errors.New("llm: ANTHROPIC_API_KEY is not set")
	}
	base := os.Getenv("ANTHROPIC_BASE_URL")
	if base == "" {
		base = "https://api.anthropic.com"
	}
	return &Anthropic{APIKey: key, BaseURL: base, Version: "2023-06-01", HTTP: http.DefaultClient}, nil
}

// Wire format of the request body.
type anthropicBlock struct {
	Type         string         `json:"type"`
	Text         string         `json:"text"`
	CacheControl map[string]any `json:"cache_control,omitempty"`
}

// anthropicContent is one content block of a message: text, tool_use or tool_result.
type anthropicContent struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`          // tool_use
	Name      string          `json:"name,omitempty"`        // tool_use
	Input     json.RawMessage `json:"input,omitempty"`       // tool_use
	ToolUseID string          `json:"tool_use_id,omitempty"` // tool_result
	Content   string          `json:"content,omitempty"`     // tool_result
	IsError   bool            `json:"is_error,omitempty"`    // tool_result
}

// anthropicMessage.Content is either a plain string or an array of blocks (json.RawMessage
// or []anthropicContent): plain text turns stay strings, exactly as before tool support.
type anthropicMessage struct {
	Role    Role `json:"role"`
	Content any  `json:"content"`
}

type anthropicTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

type anthropicRequest struct {
	Model         string             `json:"model"`
	MaxTokens     int                `json:"max_tokens"`
	System        []anthropicBlock   `json:"system,omitempty"`
	Messages      []anthropicMessage `json:"messages"`
	Tools         []anthropicTool    `json:"tools,omitempty"`
	Stream        bool               `json:"stream,omitempty"`
	Temperature   *float64           `json:"temperature,omitempty"`
	StopSequences []string           `json:"stop_sequences,omitempty"`
	OutputConfig  map[string]any     `json:"output_config,omitempty"`
}

type anthropicUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens"`
}

func (u anthropicUsage) usage() Usage { return Usage(u) }

type anthropicResponse struct {
	ID         string          `json:"id"`
	Model      string          `json:"model"`
	StopReason string          `json:"stop_reason"`
	Content    json.RawMessage `json:"content"`
	Usage      anthropicUsage  `json:"usage"`
}

// responseBlock is the subset of response block fields we read.
type responseBlock struct {
	Type  string          `json:"type"`
	Text  string          `json:"text"`
	ID    string          `json:"id"`
	Name  string          `json:"name"`
	Input json.RawMessage `json:"input"`
}

type anthropicErrorBody struct {
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func (a *Anthropic) body(req Request, stream bool) anthropicRequest {
	body := anthropicRequest{
		Model: req.Model, MaxTokens: req.MaxTokens, Stream: stream,
		Temperature: req.Temperature, StopSequences: req.StopSequences,
	}
	if req.System != "" {
		block := anthropicBlock{Type: "text", Text: req.System}
		if req.CacheSystem {
			block.CacheControl = map[string]any{"type": "ephemeral"}
		}
		body.System = []anthropicBlock{block}
	}
	for _, m := range req.Messages {
		body.Messages = append(body.Messages, anthropicMessage{Role: m.Role, Content: anthropicContentOf(m)})
	}
	for _, t := range req.Tools {
		body.Tools = append(body.Tools, anthropicTool{Name: t.Name, Description: t.Description, InputSchema: t.InputSchema})
	}
	if req.JSONSchema != nil || req.Effort != "" {
		body.OutputConfig = map[string]any{}
		if req.JSONSchema != nil {
			body.OutputConfig["format"] = map[string]any{"type": "json_schema", "schema": req.JSONSchema}
		}
		if req.Effort != "" {
			body.OutputConfig["effort"] = req.Effort
		}
	}
	return body
}

// anthropicContentOf renders one message. Order matters: in a user turn every tool_result
// block must come before any text; in an assistant turn text comes before tool_use.
func anthropicContentOf(m Message) any {
	if m.Role == RoleAssistant && len(m.Raw) > 0 {
		return m.Raw // replay verbatim: thinking blocks must come back unchanged
	}
	if len(m.ToolCalls) == 0 && len(m.ToolResults) == 0 {
		return m.Content
	}
	var blocks []anthropicContent
	for _, r := range m.ToolResults {
		blocks = append(blocks, anthropicContent{Type: "tool_result", ToolUseID: r.CallID, Content: r.Content, IsError: r.IsError})
	}
	if m.Content != "" {
		blocks = append(blocks, anthropicContent{Type: "text", Text: m.Content})
	}
	for _, c := range m.ToolCalls {
		input := c.Input
		if len(input) == 0 {
			input = json.RawMessage("{}")
		}
		blocks = append(blocks, anthropicContent{Type: "tool_use", ID: c.ID, Name: c.Name, Input: input})
	}
	return blocks
}

func (a *Anthropic) do(ctx context.Context, req Request, stream bool) (*http.Response, error) {
	payload, err := json.Marshal(a.body(req, stream))
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(a.BaseURL, "/")+"/v1/messages", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("content-type", "application/json")
	httpReq.Header.Set("x-api-key", a.APIKey)
	httpReq.Header.Set("anthropic-version", a.Version)
	if stream {
		httpReq.Header.Set("accept", "text/event-stream")
	}
	client := a.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		return nil, anthropicAPIError(resp)
	}
	return resp, nil
}

func anthropicAPIError(resp *http.Response) error {
	raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
	var eb anthropicErrorBody
	_ = json.Unmarshal(raw, &eb)
	msg := eb.Error.Message
	if msg == "" {
		msg = strings.TrimSpace(string(raw))
	}
	return &APIError{StatusCode: resp.StatusCode, Type: eb.Error.Type, Message: msg, RetryAfter: parseRetryAfter(resp.Header)}
}

// Complete sends a non-streaming request. Use it for short answers: long generations
// can outlive HTTP timeouts, and Stream is the safer default for them.
func (a *Anthropic) Complete(ctx context.Context, req Request) (*Response, error) {
	resp, err := a.do(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var ar anthropicResponse
	if err := json.NewDecoder(resp.Body).Decode(&ar); err != nil {
		return nil, fmt.Errorf("llm: decode response: %w", err)
	}
	var blocks []responseBlock
	if len(ar.Content) > 0 {
		if err := json.Unmarshal(ar.Content, &blocks); err != nil {
			return nil, fmt.Errorf("llm: decode content: %w", err)
		}
	}
	out := &Response{ID: ar.ID, Model: ar.Model, StopReason: ar.StopReason, Usage: ar.Usage.usage(), Raw: ar.Content}
	var sb strings.Builder
	for _, b := range blocks {
		switch b.Type {
		case "text":
			sb.WriteString(b.Text)
		case "tool_use":
			out.ToolCalls = append(out.ToolCalls, ToolCall{ID: b.ID, Name: b.Name, Input: b.Input})
		} // thinking, redacted_thinking, ... stay only in Raw
	}
	out.Text = sb.String()
	return out, nil
}

// streamEvent covers the fields of all Messages API stream event types we use:
// message_start, content_block_start/delta/stop, message_delta, message_stop, ping, error.
type streamEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	ContentBlock json.RawMessage `json:"content_block"`
	Message      struct {
		ID    string         `json:"id"`
		Model string         `json:"model"`
		Usage anthropicUsage `json:"usage"`
	} `json:"message"`
	Delta struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		Thinking    string `json:"thinking"`
		Signature   string `json:"signature"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage anthropicUsage `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// Stream sends a streaming request and converts SSE events into provider-neutral Events.
// Content blocks are reassembled from their deltas: a tool_use block becomes one
// EventToolCall when it stops, and the whole content array (thinking blocks included)
// is attached to EventStop as Raw so that a tool loop can replay it.
func (a *Anthropic) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		resp, err := a.do(ctx, req, true)
		if err != nil {
			yield(Event{}, err)
			return
		}
		defer resp.Body.Close()

		var blocks []*blockAcc
		for sse, err := range ReadSSE(resp.Body) {
			if err != nil {
				yield(Event{}, ctxErr(ctx, err))
				return
			}
			var ev streamEvent
			if err := json.Unmarshal([]byte(sse.Data), &ev); err != nil {
				yield(Event{}, fmt.Errorf("llm: bad stream event %q: %w", sse.Event, err))
				return
			}
			var out Event
			switch ev.Type {
			case "message_start":
				u := ev.Message.Usage.usage()
				u.OutputTokens = 0 // message_delta reports the cumulative output count; do not count twice
				out = Event{Type: EventStart, ID: ev.Message.ID, Model: ev.Message.Model, Usage: u}
			case "content_block_start":
				b := &blockAcc{start: ev.ContentBlock}
				_ = json.Unmarshal(ev.ContentBlock, &b.head)
				for len(blocks) <= ev.Index {
					blocks = append(blocks, nil)
				}
				blocks[ev.Index] = b
				continue
			case "content_block_delta":
				b := blockAt(blocks, ev.Index)
				switch ev.Delta.Type {
				case "text_delta":
					if b != nil {
						b.text.WriteString(ev.Delta.Text)
					}
					out = Event{Type: EventTextDelta, Text: ev.Delta.Text}
				case "thinking_delta":
					if b != nil {
						b.thinking.WriteString(ev.Delta.Thinking)
					}
					continue
				case "signature_delta":
					if b != nil {
						b.signature = ev.Delta.Signature
					}
					continue
				case "input_json_delta":
					if b != nil {
						b.partial.WriteString(ev.Delta.PartialJSON)
					}
					continue
				default:
					continue
				}
			case "content_block_stop":
				b := blockAt(blocks, ev.Index)
				if b == nil || b.head.Type != "tool_use" {
					continue
				}
				out = Event{Type: EventToolCall, ToolCall: ToolCall{ID: b.head.ID, Name: b.head.Name, Input: b.input()}}
			case "message_delta":
				// usage.output_tokens here is cumulative for the whole message.
				out = Event{Type: EventStop, StopReason: ev.Delta.StopReason, Usage: Usage{OutputTokens: ev.Usage.OutputTokens}}
				if raw, err := assemble(blocks); err == nil {
					out.Raw = raw
				}
			case "error":
				yield(Event{}, &APIError{StatusCode: statusForType(ev.Error.Type), Type: ev.Error.Type, Message: ev.Error.Message})
				return
			case "message_stop":
				return
			default:
				continue // ping
			}
			if !yield(out, nil) {
				return
			}
		}
	}
}

// blockAcc accumulates one streamed content block.
type blockAcc struct {
	start json.RawMessage // content_block from content_block_start
	head  struct {
		Type string `json:"type"`
		ID   string `json:"id"`
		Name string `json:"name"`
	}
	text, thinking, partial strings.Builder
	signature               string
}

func blockAt(blocks []*blockAcc, i int) *blockAcc {
	if i < 0 || i >= len(blocks) {
		return nil
	}
	return blocks[i]
}

// input returns the accumulated tool input; a tool with no parameters streams no deltas.
func (b *blockAcc) input() json.RawMessage {
	if b.partial.Len() == 0 {
		return json.RawMessage("{}")
	}
	return json.RawMessage(b.partial.String())
}

// assemble rebuilds the final content array from the streamed blocks.
func assemble(blocks []*blockAcc) (json.RawMessage, error) {
	out := make([]json.RawMessage, 0, len(blocks))
	for _, b := range blocks {
		if b == nil {
			continue
		}
		var (
			raw []byte
			err error
		)
		switch b.head.Type {
		case "text":
			raw, err = json.Marshal(map[string]string{"type": "text", "text": b.text.String()})
		case "thinking":
			raw, err = json.Marshal(map[string]string{"type": "thinking", "thinking": b.thinking.String(), "signature": b.signature})
		case "tool_use":
			raw, err = json.Marshal(map[string]any{"type": "tool_use", "id": b.head.ID, "name": b.head.Name, "input": b.input()})
		default: // redacted_thinking and other blocks arrive complete in content_block_start
			raw = b.start
		}
		if err != nil {
			return nil, err
		}
		out = append(out, raw)
	}
	return json.Marshal(out)
}

// statusForType maps an in-stream error (sent after HTTP 200) to the status it would have had.
func statusForType(t string) int {
	switch t {
	case "overloaded_error":
		return 529
	case "rate_limit_error":
		return http.StatusTooManyRequests
	case "invalid_request_error":
		return http.StatusBadRequest
	}
	return http.StatusInternalServerError
}

// ctxErr prefers the context error when a read failed because the request was canceled.
func ctxErr(ctx context.Context, err error) error {
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return err
}
