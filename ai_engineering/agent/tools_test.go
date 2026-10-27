package agent

import (
	"context"
	"encoding/json"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"algorithms_and_data_structures/ai_engineering/rag"
)

func newSandbox(t *testing.T) (*Sandbox, string) {
	t.Helper()
	base := t.TempDir()
	root := filepath.Join(base, "project")
	must(t, os.MkdirAll(filepath.Join(root, "pkg"), 0o755))
	must(t, os.MkdirAll(filepath.Join(root, ".git"), 0o755))
	must(t, os.WriteFile(filepath.Join(root, "pkg", "a.go"), []byte("package pkg\n\nfunc Search() {}\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(root, ".git", "config"), []byte("func Search secret\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(base, "secret.txt"), []byte("TOP SECRET"), 0o644))
	sb, err := NewSandbox(root)
	must(t, err)
	return sb, base
}

func must(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
}

func run(t *testing.T, tool Tool, input string) (string, error) {
	t.Helper()
	return tool.Call(context.Background(), json.RawMessage(input))
}

func TestSandboxPathTraversal(t *testing.T) {
	sb, base := newSandbox(t)
	if data, err := sb.ReadFile("pkg/a.go"); err != nil || !strings.HasPrefix(string(data), "package pkg") {
		t.Fatalf("legit read: %q %v", data, err)
	}
	if data, err := sb.ReadFile("./pkg/../pkg/a.go"); err != nil || len(data) == 0 {
		t.Fatalf("dot segments that stay inside are fine: %v", err)
	}
	for _, p := range []string{
		"../secret.txt",
		"pkg/../../secret.txt",
		"/etc/passwd",
		filepath.Join(base, "secret.txt"), // absolute path to a real file
		"..",
		"a\x00b",
	} {
		if _, err := sb.ReadFile(p); err == nil {
			t.Errorf("ReadFile(%q) escaped the sandbox", p)
		} else if !errors.Is(err, ErrOutsideSandbox) {
			t.Errorf("ReadFile(%q): want ErrOutsideSandbox, got %v", p, err)
		}
	}
	// A symlink inside the root pointing outside passes the lexical check, so os.Root must stop it.
	if err := os.Symlink(filepath.Join(base, "secret.txt"), filepath.Join(sb.Root, "link.txt")); err != nil {
		t.Skip("symlinks not supported:", err)
	}
	if data, err := sb.ReadFile("link.txt"); err == nil {
		t.Fatalf("symlink escaped the sandbox: %q", data)
	}
	if err := os.Symlink(base, filepath.Join(sb.Root, "up")); err == nil {
		if _, err := sb.ReadFile("up/secret.txt"); err == nil {
			t.Fatal("directory symlink escaped the sandbox")
		}
		if err := sb.WriteFile("up/evil.txt", []byte("x")); err == nil {
			t.Fatal("write through a symlink escaped the sandbox")
		}
	}
	if out, err := run(t, ReadFileTool(sb), `{"path":"../secret.txt"}`); err == nil || strings.Contains(out, "SECRET") {
		t.Fatalf("read_file tool: %q %v", out, err)
	}
}

func TestFileTools(t *testing.T) {
	sb, _ := newSandbox(t)
	out, err := run(t, ReadFileTool(sb), `{"path":"pkg/a.go","start_line":3,"max_lines":1}`)
	if err != nil || out != "3\tfunc Search() {}\n" {
		t.Fatalf("read_file window: %q %v", out, err)
	}
	out, err = run(t, SearchFilesTool(sb), `{"pattern":"func Search","glob":"*.go"}`)
	if err != nil || out != "pkg/a.go:3: func Search() {}\n" {
		t.Fatalf("search_files must find the definition and skip .git: %q %v", out, err)
	}
	if _, err := run(t, SearchFilesTool(sb), `{"pattern":"("}`); err == nil {
		t.Fatal("bad regexp must be an error")
	}
	out, err = run(t, ListFilesTool(sb), `{}`)
	if err != nil || out != "pkg/a.go\n" {
		t.Fatalf("list_files: %q %v", out, err)
	}
	if _, err := run(t, ListFilesTool(sb), `{"dir":"../"}`); !errors.Is(err, ErrOutsideSandbox) {
		t.Fatalf("list_files outside: %v", err)
	}
	if out, err := run(t, WriteFileTool(sb), `{"path":"new/dir/b.txt","content":"hello"}`); err != nil || !strings.Contains(out, "5 bytes") {
		t.Fatalf("write_file: %q %v", out, err)
	}
	if data, _ := sb.ReadFile("new/dir/b.txt"); string(data) != "hello" {
		t.Fatalf("written content = %q", data)
	}
}

func TestGoTestTool(t *testing.T) {
	if testing.Short() {
		t.Skip("runs the go toolchain")
	}
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("go not in PATH")
	}
	dir := t.TempDir()
	must(t, os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module sandboxed\n\ngo 1.24\n"), 0o644))
	must(t, os.WriteFile(filepath.Join(dir, "x_test.go"), []byte(`package sandboxed

import "testing"

func TestPass(t *testing.T) {}
func TestFail(t *testing.T) { t.Fatal("boom") }
`), 0o644))
	sb, _ := NewSandbox(dir)
	tool := GoTestTool(sb, 0)

	out, err := run(t, tool, `{}`)
	if err != nil || !strings.HasPrefix(out, "FAIL") || !strings.Contains(out, "boom") {
		t.Fatalf("failing tests are an observation, not an error: %q %v", out, err)
	}
	out, err = run(t, tool, `{"package":"./...","run":"TestPass"}`)
	if err != nil || !strings.HasPrefix(out, "PASS") {
		t.Fatalf("-run TestPass: %q %v", out, err)
	}
	for _, bad := range []string{
		`{"package":"../other"}`, `{"package":"./../x/..."}`, `{"package":"-exec=sh"}`,
		`{"package":"./...","run":"-v"}`, `{"package":"./...","run":"("}`,
	} {
		if _, err := run(t, tool, bad); err == nil {
			t.Errorf("%s must be rejected", bad)
		}
	}
}

func TestValidPackage(t *testing.T) {
	for p, want := range map[string]bool{
		"./...": true, "./rag": true, "./rag/...": true, "./a/b_c-d": true,
		"..": false, "./..": false, "./../x": false, "rag": false, "-run=x": false, "./a;rm": false,
	} {
		if got := validPackage(p); got != want {
			t.Errorf("validPackage(%q) = %v", p, got)
		}
	}
}

func TestRAGSearchTool(t *testing.T) {
	store := rag.NewStore(rag.HashEmbedder{Dim: 256}, &rag.Flat{})
	md := "# L17\n\n## BM25\n\nBM25 ранжирует документы по частоте терминов.\n\n## HNSW\n\nHNSW — граф малого мира для поиска соседей.\n"
	must(t, store.Add(context.Background(), rag.ChunkMarkdown("L17.md", md, 500, 50)...))
	out, err := run(t, RAGSearchTool(store), `{"query":"граф малого мира","k":1}`)
	if err != nil || !strings.Contains(out, "HNSW") || !strings.Contains(out, `file="L17.md"`) || strings.Count(out, "<passage") != 1 {
		t.Fatalf("search_docs: %q %v", out, err)
	}
}
