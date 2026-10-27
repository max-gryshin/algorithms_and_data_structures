package llm

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

var calcSpec = ToolSpec{
	Name:        "calculator",
	Description: "Evaluate an arithmetic expression.",
	InputSchema: map[string]any{"type": "object", "properties": map[string]any{"expression": map[string]any{"type": "string"}}, "required": []any{"expression"}},
}

// toolConversation is a tool loop in progress: the assistant asked for a tool, we answer.
func toolConversation(raw json.RawMessage) []Message {
	return []Message{
		{Role: RoleUser, Content: "Сколько будет 17*23?"},
		{Role: RoleAssistant, Content: "Посчитаю.", Raw: raw, ToolCalls: []ToolCall{{ID: "toolu_1", Name: "calculator", Input: json.RawMessage(`{"expression":"17*23"}`)}}},
		{Role: RoleUser, ToolResults: []ToolResult{{CallID: "toolu_1", Content: "391"}}},
	}
}

func TestAnthropicToolUseWire(t *testing.T) {
	var body struct {
		Tools []struct {
			Name        string         `json:"name"`
			InputSchema map[string]any `json:"input_schema"`
		} `json:"tools"`
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
		OutputConfig map[string]any `json:"output_config"`
	}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		json.NewDecoder(r.Body).Decode(&body)
		io.WriteString(w, `{"id":"msg_2","model":"claude-opus-5-5","stop_reason":"tool_use",
			"content":[{"type":"thinking","thinking":"","signature":"sig"},{"type":"text","text":"Ещё раз."},
			{"type":"tool_use","id":"toolu_2","name":"calculator","input":{"expression":"391+1"}}],
			"usage":{"input_tokens":50,"output_tokens":20}}`)
	}))
	defer srv.Close()

	resp, err := newAnthropic(srv.URL).Complete(context.Background(), Request{
		Model: "claude-opus-5-5", MaxTokens: 100, Tools: []ToolSpec{calcSpec}, Effort: "low",
		Messages: toolConversation(nil),
	})
	if err != nil {
		t.Fatal(err)
	}
	if len(body.Tools) != 1 || body.Tools[0].Name != "calculator" || body.Tools[0].InputSchema["type"] != "object" {
		t.Errorf("tools = %+v", body.Tools)
	}
	if body.OutputConfig["effort"] != "low" {
		t.Errorf("output_config = %v", body.OutputConfig)
	}
	if string(body.Messages[0].Content) != `"Сколько будет 17*23?"` {
		t.Errorf("plain text turn must stay a string: %s", body.Messages[0].Content)
	}
	var asst []map[string]any
	json.Unmarshal(body.Messages[1].Content, &asst)
	if len(asst) != 2 || asst[0]["type"] != "text" || asst[1]["type"] != "tool_use" || asst[1]["id"] != "toolu_1" {
		t.Errorf("assistant turn = %s", body.Messages[1].Content)
	}
	var user []map[string]any
	json.Unmarshal(body.Messages[2].Content, &user)
	if len(user) != 1 || user[0]["type"] != "tool_result" || user[0]["tool_use_id"] != "toolu_1" || user[0]["content"] != "391" {
		t.Errorf("tool result turn = %s", body.Messages[2].Content)
	}

	if resp.StopReason != "tool_use" || resp.Text != "Ещё раз." || len(resp.ToolCalls) != 1 {
		t.Fatalf("resp = %+v", resp)
	}
	if c := resp.ToolCalls[0]; c.ID != "toolu_2" || c.Name != "calculator" || string(c.Input) != `{"expression":"391+1"}` {
		t.Fatalf("call = %+v", c)
	}
	if !strings.Contains(string(resp.Raw), `"signature":"sig"`) {
		t.Fatalf("raw must keep the thinking block: %s", resp.Raw)
	}
}

func TestAnthropicReplaysRawVerbatim(t *testing.T) {
	raw := json.RawMessage(`[{"type":"thinking","thinking":"","signature":"abc"},{"type":"tool_use","id":"toolu_1","name":"calculator","input":{"expression":"17*23"}}]`)
	var got json.RawMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Messages []struct {
				Content json.RawMessage `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		got = body.Messages[1].Content
		io.WriteString(w, `{"id":"m","stop_reason":"end_turn","content":[{"type":"text","text":"391"}],"usage":{}}`)
	}))
	defer srv.Close()
	if _, err := newAnthropic(srv.URL).Complete(context.Background(), Request{Model: "m", MaxTokens: 10, Messages: toolConversation(raw)}); err != nil {
		t.Fatal(err)
	}
	if string(got) != string(raw) {
		t.Fatalf("assistant turn must be replayed byte-for-byte:\n got %s\nwant %s", got, raw)
	}
}

const anthropicToolStream = `event: message_start
data: {"type":"message_start","message":{"id":"msg_3","model":"claude-opus-5-5","usage":{"input_tokens":40,"output_tokens":1}}}|event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}|event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig0"}}|event: content_block_stop
data: {"type":"content_block_stop","index":0}|event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}|event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"Считаю."}}|event: content_block_stop
data: {"type":"content_block_stop","index":1}|event: content_block_start
data: {"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_9","name":"calculator","input":{}}}|event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{\"expre"}}|event: content_block_delta
data: {"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"ssion\": \"2+2\"}"}}|event: content_block_stop
data: {"type":"content_block_stop","index":2}|event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":30}}|event: message_stop
data: {"type":"message_stop"}`

func TestAnthropicStreamToolUse(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		writeSSE(w, strings.Split(anthropicToolStream, "|")...)
	}))
	defer srv.Close()
	resp, err := Collect(newAnthropic(srv.URL).Stream(context.Background(), Request{Model: "claude-opus-5-5", MaxTokens: 64, Tools: []ToolSpec{calcSpec}}))
	if err != nil {
		t.Fatal(err)
	}
	if resp.Text != "Считаю." || resp.StopReason != "tool_use" || len(resp.ToolCalls) != 1 {
		t.Fatalf("resp = %+v", resp)
	}
	var in struct{ Expression string }
	if err := json.Unmarshal(resp.ToolCalls[0].Input, &in); err != nil || in.Expression != "2+2" || resp.ToolCalls[0].ID != "toolu_9" {
		t.Fatalf("call = %+v (%v)", resp.ToolCalls[0], err)
	}
	var blocks []map[string]any
	if err := json.Unmarshal(resp.Raw, &blocks); err != nil || len(blocks) != 3 {
		t.Fatalf("raw = %s (%v)", resp.Raw, err)
	}
	if blocks[0]["type"] != "thinking" || blocks[0]["signature"] != "sig0" || blocks[2]["type"] != "tool_use" {
		t.Fatalf("raw blocks = %v", blocks)
	}
}

func TestOpenAIToolCalls(t *testing.T) {
	var sent []oaiMessage
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Stream   bool         `json:"stream"`
			Messages []oaiMessage `json:"messages"`
			Tools    []oaiTool    `json:"tools"`
		}
		json.NewDecoder(r.Body).Decode(&body)
		sent = body.Messages
		if len(body.Tools) != 1 || body.Tools[0].Type != "function" || body.Tools[0].Function.Name != "calculator" {
			t.Errorf("tools = %+v", body.Tools)
		}
		if !body.Stream {
			io.WriteString(w, `{"id":"c","choices":[{"message":{"content":"","tool_calls":[{"id":"call_1","type":"function","function":{"name":"calculator","arguments":"{\"expression\":\"1+1\"}"}}]},"finish_reason":"tool_calls"}]}`)
			return
		}
		writeSSE(w,
			`data: {"id":"c","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"calculator","arguments":""}}]}}]}`,
			`data: {"id":"c","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"expression\":"}}]}}]}`,
			`data: {"id":"c","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"1+1\"}"}}]},"finish_reason":"tool_calls"}]}`,
			`data: [DONE]`)
	}))
	defer srv.Close()

	c := &OpenAICompatible{BaseURL: srv.URL}
	msgs := toolConversation(nil)
	msgs[2].ToolResults[0].IsError = true
	req := Request{Model: "m", Messages: msgs, Tools: []ToolSpec{calcSpec}}
	for _, stream := range []bool{false, true} {
		var resp *Response
		var err error
		if stream {
			resp, err = Collect(c.Stream(context.Background(), req))
		} else {
			resp, err = c.Complete(context.Background(), req)
		}
		if err != nil || resp.StopReason != "tool_calls" || len(resp.ToolCalls) != 1 || string(resp.ToolCalls[0].Input) != `{"expression":"1+1"}` {
			t.Fatalf("stream=%v: %+v %v", stream, resp, err)
		}
	}
	if len(sent) != 3 || len(sent[1].ToolCalls) != 1 || sent[1].ToolCalls[0].Function.Arguments != `{"expression":"17*23"}` {
		t.Fatalf("assistant tool call on the wire: %+v", sent)
	}
	if sent[2].Role != "tool" || sent[2].ToolCallID != "toolu_1" || sent[2].Content != "ERROR: 391" {
		t.Fatalf("tool result on the wire: %+v", sent[2])
	}
}
