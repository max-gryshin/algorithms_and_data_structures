package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"sync"

	"algorithms_and_data_structures/ai_engineering/llm"
)

// Tool is a function the model can call.
type Tool interface {
	// Spec is what the model sees: name, description and input schema.
	// Treat it as prompt text — it decides when and how the model uses the tool.
	Spec() llm.ToolSpec
	// Call runs the tool. A returned error becomes an is_error tool result:
	// the model reads it and can fix its arguments, so make errors actionable.
	Call(ctx context.Context, input json.RawMessage) (string, error)
}

// Dangerous is implemented by tools with side effects that are hard to undo (writing
// files, sending messages, payments). The agent runs them only after Confirm approves.
type Dangerous interface {
	Dangerous() bool
}

// IsDangerous reports whether t asks for confirmation.
func IsDangerous(t Tool) bool {
	d, ok := t.(Dangerous)
	return ok && d.Dangerous()
}

// FuncTool adapts a typed Go function to Tool: the input schema is generated from In
// (see SchemaOf) and the raw JSON is validated and decoded into In before fn runs.
type FuncTool[In any] struct {
	name, description string
	schema            map[string]any
	fn                func(context.Context, In) (string, error)
	dangerous         bool
}

// NewTool creates a tool from a function. In must be a struct.
func NewTool[In any](name, description string, fn func(context.Context, In) (string, error)) *FuncTool[In] {
	var zero In
	return &FuncTool[In]{name: name, description: description, schema: SchemaOf(reflect.TypeOf(zero)), fn: fn}
}

// MarkDangerous makes the agent ask for confirmation before every call.
func (t *FuncTool[In]) MarkDangerous() *FuncTool[In] { t.dangerous = true; return t }

func (t *FuncTool[In]) Dangerous() bool { return t.dangerous }

func (t *FuncTool[In]) Spec() llm.ToolSpec {
	return llm.ToolSpec{Name: t.name, Description: t.description, InputSchema: t.schema}
}

func (t *FuncTool[In]) Call(ctx context.Context, input json.RawMessage) (string, error) {
	in, err := decodeInput[In](input, t.schema)
	if err != nil {
		return "", err
	}
	return t.fn(ctx, in)
}

var toolName = regexp.MustCompile(`^[a-zA-Z0-9_-]{1,64}$`)

// Registry holds the tools of an agent. Specs are returned in registration order:
// a stable tool list keeps the prompt-cache prefix (tools → system → messages) intact.
type Registry struct {
	mu    sync.RWMutex
	order []string
	tools map[string]Tool
}

// NewRegistry registers tools and panics on an invalid or duplicate name (a programming error).
func NewRegistry(tools ...Tool) *Registry {
	r := &Registry{tools: map[string]Tool{}}
	for _, t := range tools {
		if err := r.Register(t); err != nil {
			panic(err)
		}
	}
	return r
}

// Register adds a tool.
func (r *Registry) Register(t Tool) error {
	name := t.Spec().Name
	if !toolName.MatchString(name) {
		return fmt.Errorf("agent: invalid tool name %q (want %s)", name, toolName)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.tools == nil {
		r.tools = map[string]Tool{}
	}
	if _, dup := r.tools[name]; dup {
		return fmt.Errorf("agent: duplicate tool %q", name)
	}
	r.tools[name] = t
	r.order = append(r.order, name)
	return nil
}

// Get returns a tool by name.
func (r *Registry) Get(name string) (Tool, bool) {
	if r == nil {
		return nil, false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	t, ok := r.tools[name]
	return t, ok
}

// Tools returns the tools in registration order.
func (r *Registry) Tools() []Tool {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Tool, len(r.order))
	for i, n := range r.order {
		out[i] = r.tools[n]
	}
	return out
}

// Specs returns the tool specs in registration order.
func (r *Registry) Specs() []llm.ToolSpec {
	var out []llm.ToolSpec
	for _, t := range r.Tools() {
		out = append(out, t.Spec())
	}
	return out
}

// Call runs one tool call and never fails: unknown tools, invalid input, errors and
// panics all become is_error results the model can react to.
func (r *Registry) Call(ctx context.Context, call llm.ToolCall) (res llm.ToolResult) {
	res.CallID = call.ID
	t, ok := r.Get(call.Name)
	if !ok {
		res.Content, res.IsError = fmt.Sprintf("unknown tool %q", call.Name), true
		return res
	}
	defer func() {
		if p := recover(); p != nil {
			res.Content, res.IsError = fmt.Sprintf("tool %q panicked: %v", call.Name, p), true
		}
	}()
	out, err := t.Call(ctx, call.Input)
	if err != nil {
		res.Content, res.IsError = err.Error(), true
		return res
	}
	res.Content = out
	return res
}
