package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"
	"net/http"
	"strings"
)

// OpenAICompatible talks to any server implementing POST {BaseURL}/v1/chat/completions:
// vLLM, llama.cpp server, Ollama, and the inference server we build in L14.
type OpenAICompatible struct {
	APIKey  string // sent as "Authorization: Bearer ..." when non-empty
	BaseURL string
	HTTP    *http.Client
}

type oaiMessage struct {
	Role       string        `json:"role"`
	Content    string        `json:"content"`
	ToolCalls  []oaiToolCall `json:"tool_calls,omitempty"`   // assistant
	ToolCallID string        `json:"tool_call_id,omitempty"` // role "tool"
}

type oaiToolCall struct {
	Index    int    `json:"index,omitempty"` // streaming only
	ID       string `json:"id,omitempty"`
	Type     string `json:"type,omitempty"`
	Function struct {
		Name      string `json:"name,omitempty"`
		Arguments string `json:"arguments"` // a JSON object serialized as a string
	} `json:"function"`
}

type oaiTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string         `json:"name"`
		Description string         `json:"description,omitempty"`
		Parameters  map[string]any `json:"parameters"`
	} `json:"function"`
}

type oaiRequest struct {
	Model          string         `json:"model"`
	Messages       []oaiMessage   `json:"messages"`
	MaxTokens      int            `json:"max_tokens,omitempty"`
	Temperature    *float64       `json:"temperature,omitempty"`
	Stop           []string       `json:"stop,omitempty"`
	Stream         bool           `json:"stream,omitempty"`
	StreamOptions  map[string]any `json:"stream_options,omitempty"`
	ResponseFormat map[string]any `json:"response_format,omitempty"`
	Tools          []oaiTool      `json:"tools,omitempty"`
	Effort         string         `json:"reasoning_effort,omitempty"`
}

type oaiUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
}

type oaiChunk struct {
	ID      string `json:"id"`
	Model   string `json:"model"`
	Choices []struct {
		Message struct {
			Content   string        `json:"content"`
			ToolCalls []oaiToolCall `json:"tool_calls"`
		} `json:"message"`
		Delta struct {
			Content   string        `json:"content"`
			ToolCalls []oaiToolCall `json:"tool_calls"`
		} `json:"delta"`
		FinishReason *string `json:"finish_reason"`
	} `json:"choices"`
	Usage *oaiUsage `json:"usage"`
}

func (o *OpenAICompatible) do(ctx context.Context, req Request, stream bool) (*http.Response, error) {
	body := oaiRequest{Model: req.Model, MaxTokens: req.MaxTokens, Temperature: req.Temperature,
		Stop: req.StopSequences, Stream: stream, Effort: req.Effort}
	if req.System != "" {
		body.Messages = append(body.Messages, oaiMessage{Role: "system", Content: req.System})
	}
	for _, m := range req.Messages {
		body.Messages = append(body.Messages, oaiMessages(m)...)
	}
	for _, t := range req.Tools {
		var ot oaiTool
		ot.Type = "function"
		ot.Function.Name, ot.Function.Description, ot.Function.Parameters = t.Name, t.Description, t.InputSchema
		body.Tools = append(body.Tools, ot)
	}
	if stream {
		body.StreamOptions = map[string]any{"include_usage": true}
	}
	if req.JSONSchema != nil {
		body.ResponseFormat = map[string]any{"type": "json_schema",
			"json_schema": map[string]any{"name": "output", "schema": req.JSONSchema, "strict": true}}
	}
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, strings.TrimRight(o.BaseURL, "/")+"/v1/chat/completions", bytes.NewReader(payload))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("content-type", "application/json")
	if o.APIKey != "" {
		httpReq.Header.Set("authorization", "Bearer "+o.APIKey)
	}
	client := o.HTTP
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		defer resp.Body.Close()
		raw, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, &APIError{StatusCode: resp.StatusCode, Message: strings.TrimSpace(string(raw)), RetryAfter: parseRetryAfter(resp.Header)}
	}
	return resp, nil
}

func (o *OpenAICompatible) Complete(ctx context.Context, req Request) (*Response, error) {
	resp, err := o.do(ctx, req, false)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var c oaiChunk
	if err := json.NewDecoder(resp.Body).Decode(&c); err != nil {
		return nil, fmt.Errorf("llm: decode response: %w", err)
	}
	if len(c.Choices) == 0 {
		return nil, fmt.Errorf("llm: response has no choices")
	}
	out := &Response{ID: c.ID, Model: c.Model, Text: c.Choices[0].Message.Content}
	for _, tc := range c.Choices[0].Message.ToolCalls {
		out.ToolCalls = append(out.ToolCalls, ToolCall{ID: tc.ID, Name: tc.Function.Name, Input: argsJSON(tc.Function.Arguments)})
	}
	if fr := c.Choices[0].FinishReason; fr != nil {
		out.StopReason = *fr
	}
	if c.Usage != nil {
		out.Usage = Usage{InputTokens: c.Usage.PromptTokens, OutputTokens: c.Usage.CompletionTokens}
	}
	return out, nil
}

// Stream reads "data: {chunk}" events until "data: [DONE]".
func (o *OpenAICompatible) Stream(ctx context.Context, req Request) iter.Seq2[Event, error] {
	return func(yield func(Event, error) bool) {
		resp, err := o.do(ctx, req, true)
		if err != nil {
			yield(Event{}, err)
			return
		}
		defer resp.Body.Close()

		started := false
		var stop Event
		var calls []*oaiToolCall // tool calls arrive in fragments keyed by index
		for sse, err := range ReadSSE(resp.Body) {
			if err != nil {
				yield(Event{}, ctxErr(ctx, err))
				return
			}
			if sse.Data == "[DONE]" {
				break
			}
			var c oaiChunk
			if err := json.Unmarshal([]byte(sse.Data), &c); err != nil {
				yield(Event{}, fmt.Errorf("llm: bad stream chunk: %w", err))
				return
			}
			if !started {
				started = true
				if !yield(Event{Type: EventStart, ID: c.ID, Model: c.Model}, nil) {
					return
				}
			}
			if c.Usage != nil { // the final chunk (choices: []) carries usage when include_usage is set
				stop.Usage = Usage{InputTokens: c.Usage.PromptTokens, OutputTokens: c.Usage.CompletionTokens}
			}
			for _, ch := range c.Choices {
				if ch.Delta.Content != "" && !yield(Event{Type: EventTextDelta, Text: ch.Delta.Content}, nil) {
					return
				}
				for _, frag := range ch.Delta.ToolCalls {
					for len(calls) <= frag.Index {
						calls = append(calls, &oaiToolCall{})
					}
					tc := calls[frag.Index]
					if frag.ID != "" {
						tc.ID = frag.ID
					}
					tc.Function.Name += frag.Function.Name
					tc.Function.Arguments += frag.Function.Arguments
				}
				if ch.FinishReason != nil {
					stop.StopReason = *ch.FinishReason
				}
			}
		}
		for _, tc := range calls {
			ev := Event{Type: EventToolCall, ToolCall: ToolCall{ID: tc.ID, Name: tc.Function.Name, Input: argsJSON(tc.Function.Arguments)}}
			if !yield(ev, nil) {
				return
			}
		}
		stop.Type = EventStop
		yield(stop, nil)
	}
}

// oaiMessages maps one Message to Chat Completions messages: tool results become separate
// role "tool" messages (there is no is_error flag, so failures are marked in the text).
func oaiMessages(m Message) []oaiMessage {
	var out []oaiMessage
	for _, r := range m.ToolResults {
		content := r.Content
		if r.IsError {
			content = "ERROR: " + content
		}
		out = append(out, oaiMessage{Role: "tool", ToolCallID: r.CallID, Content: content})
	}
	if len(m.ToolResults) > 0 && m.Content == "" {
		return out
	}
	msg := oaiMessage{Role: string(m.Role), Content: m.Content}
	for _, c := range m.ToolCalls {
		var tc oaiToolCall
		tc.ID, tc.Type = c.ID, "function"
		tc.Function.Name, tc.Function.Arguments = c.Name, string(argsJSON(string(c.Input)))
		msg.ToolCalls = append(msg.ToolCalls, tc)
	}
	return append(out, msg)
}

// argsJSON turns a possibly empty arguments string into a JSON object.
func argsJSON(s string) json.RawMessage {
	if strings.TrimSpace(s) == "" {
		return json.RawMessage("{}")
	}
	return json.RawMessage(s)
}
