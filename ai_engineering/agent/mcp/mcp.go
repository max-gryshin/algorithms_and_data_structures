// Package mcp is a minimal Model Context Protocol server and client over the stdio
// transport (lecture L18): JSON-RPC 2.0 messages, one per line, on stdin/stdout.
//
// Implemented: initialize (with version negotiation), notifications/initialized, ping,
// tools/list, tools/call. Not implemented: resources, prompts, sampling, pagination,
// cancellation, progress, the Streamable HTTP transport.
//
// The server exposes the tools of an agent.Registry, so the same Go tools can be used by
// our own agent loop and by any MCP host (Claude Code, Claude Desktop, IDEs).
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"slices"
	"sync"

	"algorithms_and_data_structures/ai_engineering/agent"
	"algorithms_and_data_structures/ai_engineering/llm"
)

// LatestVersion is the newest protocol revision this package speaks;
// SupportedVersions lists every revision it accepts, newest first.
const LatestVersion = "2025-11-25"

var SupportedVersions = []string{"2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05"}

// JSON-RPC 2.0 error codes.
const (
	CodeParseError     = -32700
	CodeInvalidRequest = -32600
	CodeMethodNotFound = -32601
	CodeInvalidParams  = -32602
	CodeInternalError  = -32603
)

// Message is any JSON-RPC 2.0 message. A request has ID and Method, a notification has
// Method but no ID, a response has ID and Result or Error.
type Message struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"` // number or string; echoed verbatim
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// RPCError is a JSON-RPC error object.
type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

func (e *RPCError) Error() string { return fmt.Sprintf("mcp: rpc error %d: %s", e.Code, e.Message) }

// Implementation names a client or a server.
type Implementation struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// InitializeParams is sent by the client first.
type InitializeParams struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ClientInfo      Implementation `json:"clientInfo"`
}

// InitializeResult is the server's answer: the negotiated version and what it offers.
type InitializeResult struct {
	ProtocolVersion string         `json:"protocolVersion"`
	Capabilities    map[string]any `json:"capabilities"`
	ServerInfo      Implementation `json:"serverInfo"`
	Instructions    string         `json:"instructions,omitempty"`
}

// ToolInfo is one entry of tools/list.
type ToolInfo struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema map[string]any  `json:"inputSchema"`
	Annotations *ToolAnnotation `json:"annotations,omitempty"`
}

// ToolAnnotation are hints for the host's UI and permission prompts. By the spec an
// absent readOnlyHint means false and an absent destructiveHint means true, hence the
// pointers. They are self-reported by the server: a host must not trust them from an
// untrusted server.
type ToolAnnotation struct {
	ReadOnlyHint    *bool `json:"readOnlyHint,omitempty"`
	DestructiveHint *bool `json:"destructiveHint,omitempty"`
}

// Content is one item of a tool result; this package only produces text.
type Content struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// CallToolResult is the result of tools/call. Tool failures are reported here with
// IsError (the model should see them), not as JSON-RPC errors (which mean the call
// itself was malformed: unknown tool, bad params).
type CallToolResult struct {
	Content []Content `json:"content"`
	IsError bool      `json:"isError,omitempty"`
}

// Server serves a Registry over MCP.
type Server struct {
	Info         Implementation
	Instructions string // optional hint for the host's system prompt
	Tools        *agent.Registry
	// AllowDangerous exposes tools marked agent.Dangerous. Off by default: the server
	// cannot ask the user for confirmation itself, only the host can.
	AllowDangerous bool
}

func (s *Server) visible() []agent.Tool {
	var out []agent.Tool
	for _, t := range s.Tools.Tools() {
		if s.AllowDangerous || !agent.IsDangerous(t) {
			out = append(out, t)
		}
	}
	return out
}

// Serve reads newline-delimited JSON-RPC messages from in and writes responses to out
// until in reaches EOF or ctx is canceled (checked between messages). Requests are
// handled one at a time. Nothing but protocol messages may be written to out — log to
// stderr.
func (s *Server) Serve(ctx context.Context, in io.Reader, out io.Writer) error {
	r := bufio.NewReader(in)
	enc := json.NewEncoder(out) // Encode writes the value and "\n": exactly one line
	enc.SetEscapeHTML(false)
	for {
		if err := ctx.Err(); err != nil {
			return err
		}
		line, err := r.ReadBytes('\n')
		if len(bytes.TrimSpace(line)) > 0 {
			if resp := s.handle(ctx, line); resp != nil {
				if werr := enc.Encode(resp); werr != nil {
					return werr
				}
			}
		}
		if errors.Is(err, io.EOF) {
			return nil
		}
		if err != nil {
			return err
		}
	}
}

// handle processes one message and returns the response, or nil for notifications.
func (s *Server) handle(ctx context.Context, line []byte) *Message {
	var req Message
	if err := json.Unmarshal(line, &req); err != nil {
		return errorResponse(json.RawMessage("null"), CodeParseError, "parse error: "+err.Error())
	}
	if req.JSONRPC != "2.0" || req.Method == "" {
		id := req.ID
		if id == nil {
			id = json.RawMessage("null")
		}
		return errorResponse(id, CodeInvalidRequest, "invalid request")
	}
	if req.ID == nil {
		return nil // notification (notifications/initialized, notifications/cancelled, ...): no response
	}
	result, rpcErr := s.dispatch(ctx, req)
	if rpcErr != nil {
		return errorResponse(req.ID, rpcErr.Code, rpcErr.Message)
	}
	raw, err := json.Marshal(result)
	if err != nil {
		return errorResponse(req.ID, CodeInternalError, err.Error())
	}
	return &Message{JSONRPC: "2.0", ID: req.ID, Result: raw}
}

func errorResponse(id json.RawMessage, code int, msg string) *Message {
	return &Message{JSONRPC: "2.0", ID: id, Error: &RPCError{Code: code, Message: msg}}
}

func (s *Server) dispatch(ctx context.Context, req Message) (any, *RPCError) {
	switch req.Method {
	case "initialize":
		var p InitializeParams
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return nil, &RPCError{CodeInvalidParams, "initialize: " + err.Error()}
		}
		// Version negotiation: answer with the client's version if we support it,
		// otherwise with our latest; the client disconnects if it cannot speak that.
		version := LatestVersion
		if slices.Contains(SupportedVersions, p.ProtocolVersion) {
			version = p.ProtocolVersion
		}
		return InitializeResult{
			ProtocolVersion: version,
			Capabilities:    map[string]any{"tools": map[string]any{"listChanged": false}},
			ServerInfo:      s.Info,
			Instructions:    s.Instructions,
		}, nil
	case "ping":
		return struct{}{}, nil
	case "tools/list":
		tools := []ToolInfo{}
		for _, t := range s.visible() {
			spec := t.Spec()
			info := ToolInfo{Name: spec.Name, Description: spec.Description, InputSchema: spec.InputSchema}
			destructive := agent.IsDangerous(t)
			info.Annotations = &ToolAnnotation{DestructiveHint: &destructive}
			tools = append(tools, info)
		}
		return map[string]any{"tools": tools}, nil
	case "tools/call":
		var p struct {
			Name      string          `json:"name"`
			Arguments json.RawMessage `json:"arguments"`
		}
		if err := json.Unmarshal(req.Params, &p); err != nil {
			return nil, &RPCError{CodeInvalidParams, "tools/call: " + err.Error()}
		}
		if !slices.ContainsFunc(s.visible(), func(t agent.Tool) bool { return t.Spec().Name == p.Name }) {
			return nil, &RPCError{CodeInvalidParams, fmt.Sprintf("unknown tool: %s", p.Name)}
		}
		res := s.Tools.Call(ctx, llm.ToolCall{ID: string(req.ID), Name: p.Name, Input: p.Arguments})
		return CallToolResult{Content: []Content{{Type: "text", Text: res.Content}}, IsError: res.IsError}, nil
	}
	return nil, &RPCError{CodeMethodNotFound, "method not found: " + req.Method}
}

// Client is a minimal MCP client (the part of a host that talks to one server).
// Calls are sequential; it is safe for concurrent use but does not pipeline.
type Client struct {
	mu   sync.Mutex
	r    *bufio.Reader
	enc  *json.Encoder
	next int
}

// NewClient talks to a server that reads w and writes r (e.g. a subprocess's stdin/stdout).
func NewClient(r io.Reader, w io.Writer) *Client {
	return &Client{r: bufio.NewReader(r), enc: json.NewEncoder(w)}
}

// Call sends a request and decodes the result into out (if non-nil).
func (c *Client) Call(ctx context.Context, method string, params, out any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return err
	}
	c.next++
	id := json.RawMessage(fmt.Sprint(c.next))
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	if err := c.enc.Encode(Message{JSONRPC: "2.0", ID: id, Method: method, Params: raw}); err != nil {
		return err
	}
	for {
		line, err := c.r.ReadBytes('\n')
		if err != nil {
			return fmt.Errorf("mcp: read: %w", err)
		}
		var msg Message
		if err := json.Unmarshal(line, &msg); err != nil {
			return fmt.Errorf("mcp: bad message: %w", err)
		}
		if msg.Method != "" || !bytes.Equal(msg.ID, id) {
			continue // a server notification or request; not supported by this client
		}
		if msg.Error != nil {
			return msg.Error
		}
		if out != nil {
			return json.Unmarshal(msg.Result, out)
		}
		return nil
	}
}

// Notify sends a notification (no response expected).
func (c *Client) Notify(method string, params any) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	raw, err := json.Marshal(params)
	if err != nil {
		return err
	}
	return c.enc.Encode(Message{JSONRPC: "2.0", Method: method, Params: raw})
}

// Initialize performs the handshake: initialize → result → notifications/initialized.
func (c *Client) Initialize(ctx context.Context, info Implementation) (*InitializeResult, error) {
	var res InitializeResult
	err := c.Call(ctx, "initialize", InitializeParams{ProtocolVersion: LatestVersion, Capabilities: map[string]any{}, ClientInfo: info}, &res)
	if err != nil {
		return nil, err
	}
	if !slices.Contains(SupportedVersions, res.ProtocolVersion) {
		return nil, fmt.Errorf("mcp: server speaks unsupported protocol version %q", res.ProtocolVersion)
	}
	return &res, c.Notify("notifications/initialized", map[string]any{})
}

// ListTools returns the server's tools.
func (c *Client) ListTools(ctx context.Context) ([]ToolInfo, error) {
	var res struct {
		Tools []ToolInfo `json:"tools"`
	}
	err := c.Call(ctx, "tools/list", map[string]any{}, &res)
	return res.Tools, err
}

// CallTool calls a tool with JSON arguments.
func (c *Client) CallTool(ctx context.Context, name string, args json.RawMessage) (*CallToolResult, error) {
	var res CallToolResult
	if len(args) == 0 {
		args = json.RawMessage("{}")
	}
	err := c.Call(ctx, "tools/call", map[string]any{"name": name, "arguments": args}, &res)
	return &res, err
}

// RemoteTools wraps the server's tools as agent.Tools, so our agent loop can use any
// MCP server — this is what a host does. Unless trustHints is true, every remote tool
// is treated as Dangerous (the agent asks Confirm before each call); with trustHints,
// tools the server declares read-only or non-destructive run without confirmation.
func RemoteTools(ctx context.Context, c *Client, trustHints bool) ([]agent.Tool, error) {
	infos, err := c.ListTools(ctx)
	if err != nil {
		return nil, err
	}
	out := make([]agent.Tool, len(infos))
	for i, info := range infos {
		out[i] = &remoteTool{c: c, info: info, trust: trustHints}
	}
	return out, nil
}

type remoteTool struct {
	c     *Client
	info  ToolInfo
	trust bool
}

func (t *remoteTool) Spec() llm.ToolSpec {
	return llm.ToolSpec{Name: t.info.Name, Description: t.info.Description, InputSchema: t.info.InputSchema}
}

func (t *remoteTool) Dangerous() bool {
	a := t.info.Annotations
	if !t.trust || a == nil {
		return true
	}
	readOnly := a.ReadOnlyHint != nil && *a.ReadOnlyHint
	destructive := a.DestructiveHint == nil || *a.DestructiveHint
	return !readOnly && destructive
}

func (t *remoteTool) Call(ctx context.Context, input json.RawMessage) (string, error) {
	res, err := t.c.CallTool(ctx, t.info.Name, input)
	if err != nil {
		return "", err
	}
	var b bytes.Buffer
	for _, c := range res.Content {
		if c.Type == "text" {
			b.WriteString(c.Text)
		}
	}
	if res.IsError {
		return "", errors.New(b.String())
	}
	return b.String(), nil
}
