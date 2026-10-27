package llm

import "encoding/json"

// Function calling (lecture L18). The model never executes anything: it generates a
// structured request "call tool X with input Y", the API returns it as a separate content
// block, and *our* code runs the function and sends the result back in the next turn.

// ToolSpec describes a function the model may call. InputSchema is a JSON Schema
// object ({"type": "object", "properties": ..., "required": ...}).
type ToolSpec struct {
	Name        string
	Description string
	InputSchema map[string]any
}

// ToolCall is the model's request to run a tool: an Anthropic tool_use block or an
// OpenAI tool_calls item. ID links the call to its ToolResult.
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage // a JSON object; parse it, never string-match it
}

// ToolResult answers one ToolCall. IsError tells the model the call failed, so it can
// retry with different arguments or choose another approach.
type ToolResult struct {
	CallID  string
	Content string
	IsError bool
}
