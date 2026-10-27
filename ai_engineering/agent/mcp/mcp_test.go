package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"io"
	"iter"
	"strings"
	"testing"

	"algorithms_and_data_structures/ai_engineering/agent"
	"algorithms_and_data_structures/ai_engineering/llm"
)

// connect runs a server on in-memory pipes and returns a client wired to it.
func connect(t *testing.T, s *Server) *Client {
	t.Helper()
	c2sR, c2sW := io.Pipe() // client → server (server's stdin)
	s2cR, s2cW := io.Pipe() // server → client (server's stdout)
	done := make(chan error, 1)
	go func() {
		err := s.Serve(context.Background(), c2sR, s2cW)
		s2cW.Close()
		done <- err
	}()
	t.Cleanup(func() {
		c2sW.Close() // EOF on the server's stdin ends Serve
		if err := <-done; err != nil {
			t.Errorf("Serve: %v", err)
		}
	})
	return NewClient(s2cR, c2sW)
}

func testServer(t *testing.T) *Server {
	sb, err := agent.NewSandbox(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	reg := agent.NewRegistry(append([]agent.Tool{agent.Calculator()}, agent.ProjectTools(sb, true)...)...)
	return &Server{Info: Implementation{Name: "algo", Version: "0.1.0"}, Tools: reg, Instructions: "Go course tools."}
}

func TestHandshakeListCall(t *testing.T) {
	c := connect(t, testServer(t))
	ctx := context.Background()

	init, err := c.Initialize(ctx, Implementation{Name: "test-host", Version: "1"})
	if err != nil {
		t.Fatal(err)
	}
	if init.ProtocolVersion != LatestVersion || init.ServerInfo.Name != "algo" || init.Capabilities["tools"] == nil {
		t.Fatalf("initialize = %+v", init)
	}
	if err := c.Call(ctx, "ping", map[string]any{}, nil); err != nil {
		t.Fatalf("ping: %v", err)
	}

	tools, err := c.ListTools(ctx)
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, ti := range tools {
		names = append(names, ti.Name)
		if ti.InputSchema["type"] != "object" {
			t.Errorf("%s: inputSchema = %v", ti.Name, ti.InputSchema)
		}
	}
	// write_file is Dangerous and AllowDangerous is off: not exposed.
	if got := strings.Join(names, ","); got != "calculator,list_files,read_file,search_files,go_test" {
		t.Fatalf("tools = %s", got)
	}

	res, err := c.CallTool(ctx, "calculator", json.RawMessage(`{"expression":"2^10"}`))
	if err != nil || res.IsError || res.Content[0].Type != "text" || res.Content[0].Text != "1024" {
		t.Fatalf("call = %+v %v", res, err)
	}
	// A tool failure is a successful RPC with isError, so the model sees it.
	res, err = c.CallTool(ctx, "calculator", json.RawMessage(`{"expression":"1/0"}`))
	if err != nil || !res.IsError || !strings.Contains(res.Content[0].Text, "division by zero") {
		t.Fatalf("tool error = %+v %v", res, err)
	}
	res, err = c.CallTool(ctx, "read_file", json.RawMessage(`{"path":"../../etc/passwd"}`))
	if err != nil || !res.IsError || !strings.Contains(res.Content[0].Text, "outside the sandbox") {
		t.Fatalf("traversal over MCP = %+v %v", res, err)
	}
	// Unknown (or hidden) tools are protocol errors.
	var rpcErr *RPCError
	if _, err := c.CallTool(ctx, "write_file", json.RawMessage(`{"path":"x","content":"y"}`)); !errors.As(err, &rpcErr) || rpcErr.Code != CodeInvalidParams {
		t.Fatalf("hidden tool: %v", err)
	}
	if err := c.Call(ctx, "resources/list", map[string]any{}, nil); !errors.As(err, &rpcErr) || rpcErr.Code != CodeMethodNotFound {
		t.Fatalf("unknown method: %v", err)
	}
}

// TestWireFormat drives the server with raw lines, the way Claude Code does over stdio.
func TestWireFormat(t *testing.T) {
	in := strings.Join([]string{
		`{"jsonrpc":"2.0","id":"a","method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"x","version":"1"}}}`,
		`{"jsonrpc":"2.0","method":"notifications/initialized"}`,
		`{not json`,
		`{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"calculator","arguments":{"expression":"6*7"}}}`,
		`{"jsonrpc":"2.0","id":8,"method":"initialize","params":{"protocolVersion":"1999-01-01","capabilities":{},"clientInfo":{"name":"x","version":"1"}}}`,
	}, "\n") + "\n"
	var out strings.Builder
	if err := testServer(t).Serve(context.Background(), strings.NewReader(in), &out); err != nil {
		t.Fatal(err)
	}
	var lines []Message
	sc := bufio.NewScanner(strings.NewReader(out.String()))
	for sc.Scan() {
		var m Message
		if err := json.Unmarshal(sc.Bytes(), &m); err != nil {
			t.Fatalf("every output line must be one JSON message: %q", sc.Text())
		}
		lines = append(lines, m)
	}
	// 4 responses: the notification gets none.
	if len(lines) != 4 {
		t.Fatalf("got %d responses:\n%s", len(lines), out.String())
	}
	if string(lines[0].ID) != `"a"` || !strings.Contains(string(lines[0].Result), `"protocolVersion":"2025-06-18"`) {
		t.Errorf("string id echoed and client version accepted: %+v", lines[0])
	}
	if lines[1].Error == nil || lines[1].Error.Code != CodeParseError || string(lines[1].ID) != "null" {
		t.Errorf("parse error: %+v", lines[1])
	}
	if string(lines[2].ID) != "7" || string(lines[2].Result) != `{"content":[{"type":"text","text":"42"}]}` {
		t.Errorf("tools/call: id=%s result=%s", lines[2].ID, lines[2].Result)
	}
	if !strings.Contains(string(lines[3].Result), `"protocolVersion":"`+LatestVersion+`"`) {
		t.Errorf("unknown client version → server's latest: %s", lines[3].Result)
	}
}

// TestRemoteToolsInAgent closes the loop: our agent uses tools served over MCP.
func TestRemoteToolsInAgent(t *testing.T) {
	c := connect(t, testServer(t))
	ctx := context.Background()
	if _, err := c.Initialize(ctx, Implementation{Name: "go-agent", Version: "1"}); err != nil {
		t.Fatal(err)
	}
	remote, err := RemoteTools(ctx, c, true)
	if err != nil {
		t.Fatal(err)
	}
	step := 0
	model := modelFunc(func(req llm.Request) (*llm.Response, error) {
		step++
		if step == 1 {
			return &llm.Response{StopReason: "tool_use", ToolCalls: []llm.ToolCall{{ID: "t1", Name: "calculator", Input: json.RawMessage(`{"expression":"17*23"}`)}}}, nil
		}
		return &llm.Response{StopReason: "end_turn", Text: "= " + req.Messages[2].ToolResults[0].Content}, nil
	})
	a := &agent.Agent{Client: model, Tools: agent.NewRegistry(remote...)}
	res, err := a.Run(ctx, "17*23?")
	if err != nil || res.Text != "= 391" {
		t.Fatalf("%+v %v", res, err)
	}

	untrusted, _ := RemoteTools(ctx, c, false)
	if !agent.IsDangerous(untrusted[0]) || agent.IsDangerous(remote[0]) {
		t.Fatal("without trusted hints every remote tool needs confirmation")
	}
}

type modelFunc func(llm.Request) (*llm.Response, error)

func (f modelFunc) Complete(_ context.Context, req llm.Request) (*llm.Response, error) { return f(req) }

func (f modelFunc) Stream(ctx context.Context, req llm.Request) iter.Seq2[llm.Event, error] {
	return func(yield func(llm.Event, error) bool) {
		resp, err := f(req)
		if err != nil {
			yield(llm.Event{}, err)
			return
		}
		yield(llm.Event{Type: llm.EventStop, StopReason: resp.StopReason}, nil)
	}
}
